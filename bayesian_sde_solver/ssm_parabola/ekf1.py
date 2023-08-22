import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from bayesian_sde_solver.ssm_parabola.ekf import _solver


def solver(key, init, delta, drift, diffusion, h, N, sqrt=False, theta=0.0):
    """
    Wrapper for EKF1 with new prior at each step.
    Including Brownian increments and Levy areas as part of the observation.
    Using sqrt.
    """
    dim = init.shape[0]

    filtered = _solver(init, drift, diffusion, delta, h, N, sqrt, EKF0=False, theta=theta)
    m, P = filtered
    if key is not None:
        sqrtP = P if sqrt else jnp.real(linalg.sqrtm(P))
        sample = m + sqrtP @ jax.random.multivariate_normal(key, jnp.zeros((4 * dim,)), jnp.eye(4 * dim))
    else:
        sample = m  # no sampling, only mean.

    sample_point = jnp.vstack(sample[::4]).reshape((dim,))
    sample_brownian = jnp.vstack(sample[2::4]).reshape((dim,))
    sample_propto_levy_area = jnp.vstack(sample[3::4]).reshape((dim,))
    return sample_point, sample_brownian, sample_propto_levy_area
