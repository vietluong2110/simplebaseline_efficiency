import numpy as np
from math import sqrt
from flax import nnx
import jax
import jax.numpy as jnp
import pdb
import pywt
from jax import lax
class JAX_WaveletEmbedding(nnx.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2, kernel_size=None):
        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = jnp.array(self.wavelet.dec_lo[::-1], dtype=jnp.float64)
                h1 = jnp.array(self.wavelet.dec_hi[::-1], dtype=jnp.float64)
            else:
                h0 = jnp.array(self.wavelet.rec_lo[::-1], dtype=jnp.float64)
                h1 = jnp.array(self.wavelet.rec_hi[::-1], dtype=jnp.float64)
            self.h0 = nnx.Param(jnp.tile(h0[None, None, :], [self.d_channel, 1, 1]))
            self.h1 = nnx.Param(jnp.tile(h1[None, None, :], [self.d_channel, 1, 1]))
            self.kernel_size = self.h0.shape[-1]
        
    def __call__(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = jnp.pad(approx_coeffs, ((0, 0), (0, 0), pad), "wrap")
            # pdb.set_trace()
            detail_coeff = lax.conv_general_dilated(
                    lhs=approx_coeffs_pad,
                    rhs=h1.value,
                    window_strides=(1,),
                    padding='VALID',
                    rhs_dilation=(dilation,),
                    dimension_numbers=('NCH', 'OIH', 'NCH'),
                    feature_group_count=x.shape[1]
                )
            approx_coeffs = lax.conv_general_dilated(
                    lhs=approx_coeffs_pad,
                    rhs=h0.value,
                    window_strides=(1,),
                    padding='VALID',
                    rhs_dilation=(dilation,),
                    dimension_numbers=('NCH', 'OIH', 'NCH'),
                    feature_group_count=x.shape[1]
                )
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)
        result = jnp.stack(list(reversed(coeffs)), axis=-2)
        return result

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        # m = coeffs.shape[-2] - 1
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        # pdb.set_trace()
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = jnp.pad(approx_coeff, ((0, 0), (0, 0), pad), "wrap")
            detail_coeff_pad = jnp.pad(detail_coeff, ((0, 0), (0, 0), pad), "wrap")
            y = lax.conv_general_dilated(
                lhs=approx_coeff_pad,
                rhs=g0.value,
                window_strides=(1,),
                padding='VALID',               # Already padded manually
                rhs_dilation=(dilation,),      # Dilation factor in the kernel
                dimension_numbers=('NCH','OIH','NCH'),
                feature_group_count=approx_coeff.shape[1])  + \
            lax.conv_general_dilated(
                lhs=detail_coeff_pad,
                rhs=g1.value,
                window_strides=(1,),
                padding='VALID',               # Already padded manually
                rhs_dilation=(dilation,),      # Dilation factor in the kernel
                dimension_numbers=('NCH','OIH','NCH'),
                feature_group_count=detail_coeff.shape[1]
            )
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff

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
        print(kernel_size)
        self.out_projection = nnx.Sequential(
            nnx.Linear(in_features = d_model, out_features = d_model, kernel_init = kernel_init, bias_init = bias_init, rngs = rngs),
            JAX_WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )
        
    def __call__(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # pdb.set_trace()
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
        # pdb.set_trace()
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