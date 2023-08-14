import jax.numpy as jnp
from numpy.typing import ArrayLike
from typing import Tuple
from math import sqrt

def interlace(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return jnp.vstack((x, y)).reshape((-1,), order='F')


def multiple_interlace(arrays: Tuple[ArrayLike]) -> ArrayLike:
    return jnp.vstack(arrays).reshape((-1,), order='F')


def interlace_matrix(v: ArrayLike, w: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    dim = v.shape[0]
    var = jnp.block([[v, w],
                     [x, y]]
                    )
    _var = var.copy()
    # interlace the variance matrix
    _var.at[::2, ::2].set(var.at[:dim, :dim].get())
    _var.at[1::2, 1::2].set(var.at[dim:, dim:].get())
    _var.at[1::2, ::2].set(var.at[dim:, :dim].get())
    _var.at[::2, 1::2].set(var.at[:dim, dim:].get())
    return _var

def multiple_interlace_matrix(arrays: Tuple[ArrayLike]) -> ArrayLike:
    dim = arrays[0].shape[0]
    n = int(sqrt(len(arrays)))
    var = jnp.block([[arrays[i*n+j] for j in range(n)] for i in range(n)])
    _var = var.copy()
    raise NotImplementedError