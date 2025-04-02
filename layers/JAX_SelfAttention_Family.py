import numpy as np
from math import sqrt
from flax import nnx
import jax
import jax.numpy as jnp
import pdb
import pywt
from jax import lax
import jaxwt as jwt
class JAX_WaveletEmbedding(nnx.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2, kernel_size=None):
        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            self.wavelet = wv
        
    def __call__(self, x):
        if self.swt:
            coeffs = jwt.swt(x, self.wavelet, level = self.m)
        else:
            coeffs = jwt.iswt(x, self.wavelet)
        return coeffs


class JAX_GeomAttentionLayer(nnx.Module):
    def __init__(self, attention, d_model, rngs: nnx.Rngs,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5,): #sym5
        self.d_channel = d_channel
        self.inner_attention = attention

        kernel_init = jax.nn.initializers.variance_scaling(scale = 1, mode = 'fan_in', distribution = 'uniform')
        bias_init = jax.nn.initializers.normal()

        self.swt = JAX_WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.query_projection = nnx.Sequential(
            nnx.Linear(in_features = d_model, out_features = d_model, kernel_init = kernel_init, bias_init = bias_init, rngs = rngs),
            nnx.Dropout(geomattn_dropout, rngs = rngs)
            
        )
        self.key_projection = nnx.Sequential(
            nnx.Linear(in_features = d_model, out_features = d_model, kernel_init = kernel_init, bias_init = bias_init, rngs = rngs),
            nnx.Dropout(geomattn_dropout, rngs = rngs)
        )
        self.value_projection = nnx.Sequential(
            nnx.Linear(in_features = d_model, out_features = d_model, kernel_init = kernel_init, bias_init = bias_init, rngs = rngs),
            nnx.Dropout(geomattn_dropout, rngs = rngs)
        )
        self.out_projection = nnx.Sequential(
            nnx.Linear(in_features = d_model, out_features = d_model, kernel_init = kernel_init, bias_init = bias_init, rngs = rngs),
            JAX_WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )
        
    def __call__(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)
        queries = jnp.permute_dims(self.query_projection(queries), (0,3,2,1))
        keys = jnp.permute_dims(self.key_projection(keys), (0,3,2,1))
        values = jnp.permute_dims(self.value_projection(values), (0,3,2,1))
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = self.out_projection(jnp.permute_dims(out, (0,3,2,1)))
        return out, attn

class JAX_GeomAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, 
                 output_attention=False,
                 alpha=1.,):
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nnx.Dropout(attention_dropout, rngs = rngs)
        
        self.alpha = alpha 

    def __call__(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute the dot product attention
        dot_product = jnp.einsum("blhe,bshe->bhls", queries, keys)

        # Compute squared norms of queries and keys
        queries_norm2 = jnp.sum(queries**2, axis=-1)
        keys_norm2 = jnp.sum(keys**2, axis=-1)
        # Reshape norms for broadcasting
        queries_norm2 = jnp.expand_dims(jnp.permute_dims(queries_norm2, (0, 2, 1)) , -1)  # (B, H, L, 1)
        keys_norm2 = jnp.expand_dims(jnp.permute_dims(keys_norm2, (0, 2, 1)) , -1)        # (B, H, 1, S)
        # Compute squared norm of the wedge product
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2  # (B, H, L, S)
        wedge_norm2 = jax.nn.relu(wedge_norm2)
        # Compute the wedge norm
        wedge_norm = jnp.sqrt(wedge_norm2 + 1e-8)

        # Combined attention score
        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = jnp.tril(jnp.ones(L, S))
            masked = (attn_mask[:, None, None, :] == 0)  # shape extended by two dims
            scores = jnp.where(masked, -jnp.inf, scores)

        A = self.dropout(jax.nn.softmax(scores, axis=-1)) 

        V = jnp.einsum("bhls,bshd->blhd", A, values)
        
        return (V, jnp.mean(jnp.abs(scores)))