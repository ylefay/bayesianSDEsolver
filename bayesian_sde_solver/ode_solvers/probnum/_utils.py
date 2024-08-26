from typing import Tuple

import jax.numpy as jnp
from numpy.typing import ArrayLike


def interlace(arrays: Tuple[ArrayLike]) -> ArrayLike:
    """
    Interlace arrays, e.g interlace(([a, b, c], [d, e, f])) = [a, d, b, e, c, f].
    """
    return jnp.vstack(arrays).reshape((-1,), order='F')


def interlace_matrix(v: ArrayLike, w: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    From 4 block matrices corresponding to covariances matrices between two vectors,
    construct the interlaced covariance matrix between the two interlaced vectors.
    """
    dim = v.shape[0]
    var = jnp.block([[v, w],
                     [x, y]]
                    )
    _var = var.copy()
    # interlace the variance matrix
    _var = _var.at[::2, ::2].set(var.at[:dim, :dim].get())
    _var = _var.at[1::2, 1::2].set(var.at[dim:, dim:].get())
    _var = _var.at[1::2, ::2].set(var.at[dim:, :dim].get())
    _var = _var.at[::2, 1::2].set(var.at[:dim, dim:].get())
    return _var
