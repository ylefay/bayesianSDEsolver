import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


def mle_diffusion(zs: ArrayLike, Ss: ArrayLike):
    """
    Maximum likelihood estimate for the diffusion coefficient on one coordinate
    zs, Ss sequence of observations, means and covariances for the measurement error
    """
    N = zs.shape[0]
    dim = zs.shape[1]

    @jax.vmap
    def _sum(z, S):
        temp = jax.scipy.linalg.solve(S, z)
        return jnp.dot(z, temp)
        #temp = jax.scipy.linalg.solve_triangular(S, z, lower=True) #assuming S is cholesky
        #return jnp.dot(temp, temp)

    mle = jnp.sum(_sum(zs, Ss)) / (N * dim)  # Assuming we scale the variance of the d-dimensional problem by the same coeff. then we must divide by d
    return mle
