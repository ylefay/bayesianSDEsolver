import jax.numpy as jnp


def interlace(x, y):
    return jnp.vstack((x, y)).reshape((-1,), order='F')

