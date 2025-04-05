
import pdb
from flax import nnx
import jax.numpy as jnp
import jax 

class JAX_Encoder(nnx.Module):
    def __init__(self, attn_layers, norm_layer = None):
        self.attn_layers = attn_layers
        self.norm = norm_layer

    def __call__(self, x, attn_mask = None, tau = None, delta = None): 
        attns = [] 
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau = tau, delta = delta)
            attns.append(attn)
        if self.norm is not None: 
            x = self.norm(x)
        
        return x, attns

class JAX_EncoderLayer(nnx.Module):
    def __init__(self, attention, d_model, rngs: nnx.Rngs, d_ff=None, dropout=0.1, activation="relu", dec_in=866):
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        kernel_init = jax.nn.initializers.variance_scaling(scale = 1, mode = 'fan_in', distribution = 'uniform')
        bias_init = jax.nn.initializers.normal()
        self.conv1 = nnx.Conv(in_features=d_model, out_features=d_ff, kernel_size=(1, ),
                               kernel_init = kernel_init, bias_init = bias_init, rngs = rngs)
        self.conv2 = nnx.Conv(in_features=d_ff, out_features=d_model, kernel_size=(1, ), 
                                kernel_init = kernel_init, bias_init = bias_init, rngs = rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs = rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs = rngs)
        self.dropout = nnx.Dropout(dropout, rngs = rngs)
        self.activation = nnx.relu if activation == "relu" else nnx.gelu
        
    def __call__(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # pdb.set_trace()
        new_x = jnp.asarray(new_x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x + y), attn