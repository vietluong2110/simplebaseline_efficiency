import jax
import jax.numpy as jnp
from jax import lax

def f(x):
  print(jnp.arange(x))
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jax.jit(f, static_argnames='x')

print(f(10))