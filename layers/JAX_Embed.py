
from flax import nnx
import jax.numpy as jnp 
import pdb
import jax 

class JAX_DataEmbedding_inverted(nnx.Module):
    def __init__(self, c_in, d_model, rngs, dropout = 0.1):
        kernel_init = jax.nn.initializers.variance_scaling(scale = 1, mode = 'fan_in', distribution = 'uniform')
        bias_init = jax.nn.initializers.normal()
        self.value_embedding = nnx.Linear(in_features = c_in, out_features = d_model, kernel_init=kernel_init, bias_init = bias_init,  rngs = rngs)
        self.dropout = nnx.Dropout(dropout, rngs = rngs)

    def __call__(self, x, x_mark):
        x = jnp.permute_dims(x, (0, 2, 1))
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(jnp.concat([x, jnp.permute_dims(x_mark, (0, 2, 1))], 1))
        
        return self.dropout(x)