import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


def mle_diffusion(ys: ArrayLike, ms: ArrayLike, Ps: ArrayLike):
    """
    Maximum likelihood estimate for the diffusion coefficient on one coordinate
    ys, ms, Ps sequence of observations, means and covariances
    """
    N = ys.shape[0]

    @jax.vmap
    def _sum(y, m, P):
        return (y - m).T @ jnp.linalg.pinv(P) @ (y - m)

    mle = jnp.sum(_sum(ys, ms, Ps)) / N
    return mle
