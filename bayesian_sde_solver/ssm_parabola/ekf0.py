import jax
import jax.numpy as jnp

from bayesian_sde_solver.ode_solvers.probnum import multiple_interlace
from bayesian_sde_solver.ssm_parabola.ekf import _solver


def solver(key, init, delta, drift, diffusion, h, N, sqrt=True):
    """
    Wrapper for EKF1 with new prior at each step.
    Including Brownian increments and Levy areas as part of the observation.
    Using sqrt.
    """
    x_init = init[::3]
    brownian_increments_init = init[1::3]
    second_coeff_pol = init[2::3]
    vector_field_init = drift(x_init, 0.0) + diffusion(x_init, 0.0) @ (
                brownian_increments_init - jnp.sqrt(6)*second_coeff_pol) / delta
    dim = x_init.shape[0]
    # Zero initial variance
    init = (
        multiple_interlace((x_init, vector_field_init, brownian_increments_init, second_coeff_pol)),  # ...
        jnp.zeros((4 * dim, 4 * dim))
    )

    filtered = _solver(init, drift, diffusion, delta, h, N, sqrt, EKF0=True)
    m, P = filtered
    if key is not None:
        if not sqrt:
            cholP = jnp.linalg.cholesky(P)
        else:
            cholP = P
        last_sample = m + cholP @ jax.random.multivariate_normal(key, jnp.zeros((4 * dim,)), jnp.eye(4 * dim))
        return jnp.vstack(last_sample[::4]).reshape((dim,))
    last_value = jnp.vstack(m[::4]).reshape((dim,))
    return last_value  # return only the mean
