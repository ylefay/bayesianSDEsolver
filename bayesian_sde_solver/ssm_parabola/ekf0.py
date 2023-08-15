import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from bayesian_sde_solver.ssm_parabola.ekf import _solver


def solver(key, init, delta, drift, diffusion, h, N, sqrt=False):
    """
    Wrapper for EKF1 with new prior at each step.
    Including Brownian increments and Levy areas as part of the observation.
    Using sqrt.
    """
    dim = init.shape[0]

    filtered = _solver(init, drift, diffusion, delta, h, N, sqrt, EKF0=True)
    m, P = filtered
    if key is not None:
        if not sqrt:
            sqrtP = jnp.real(linalg.sqrtm(P))
        else:
            sqrtP = P
        last_sample = m + sqrtP @ jax.random.multivariate_normal(key, jnp.zeros((4 * dim,)), jnp.eye(4 * dim))
        return jnp.vstack(last_sample[::4]).reshape((dim,))
    last_value = jnp.vstack(m[::4]).reshape((dim,))
    return last_value  # return only the mean
