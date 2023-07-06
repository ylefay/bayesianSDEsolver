import jax.numpy as jnp
from numpy.typing import ArrayLike


def interlace(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return jnp.vstack((x, y)).reshape((-1,), order='F')
