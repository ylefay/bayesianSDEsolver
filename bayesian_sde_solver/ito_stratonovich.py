from typing import Callable

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


def drift_correction(diffusion: Callable, t: float, x: ArrayLike) -> ArrayLike:
    """
    Computes the drift correction term for the Stratonovich-Ito conversion.
    See Kloeden, Platen, 1999, chapter 4.9.
    """
    diff_val = diffusion(x, t)
    jac_val = jax.jacfwd(lambda z: diffusion(z, t))(x)
    return jnp.einsum("jk,ikj->i", diff_val, jac_val)


def to_stratonovich(drift, diffusion):
    def new_drift(x, t):
        return drift(x, t) - 0.5 * drift_correction(diffusion, t, x)

    return new_drift, diffusion


def to_ito(drift, diffusion):
    def new_drift(x, t):
        return drift(x, t) + 0.5 * drift_correction(diffusion, t, x)

    return new_drift, diffusion
