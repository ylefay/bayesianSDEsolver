import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from bayesian_sde_solver.ode_solvers.ekf import _solver
from bayesian_sde_solver.ode_solvers.probnum import interlace


def solver(key, init, vector_field, h, N, sqrt=True, theta=0.0):
    """
    Wrapper for EKF1 with new prior at each step.
    """
    dim = init.shape[0]
    # Zero initial variance
    init = (
        interlace((init, vector_field(init, 0.0))),
        jnp.zeros((2 * dim, 2 * dim))
    )
    filtered = _solver(init, vector_field, h, N, sqrt, EKF0=False, theta=theta)
    m, P = filtered
    if key is not None:
        sqrtP = P if sqrt else jnp.real(linalg.sqrtm(P))
        last_sample = m + sqrtP @ jax.random.multivariate_normal(key, jnp.zeros((2 * dim,)), jnp.eye(2 * dim))
        return jnp.vstack(last_sample[::2]).reshape((dim,))
    last_value = jnp.vstack(m[::2]).reshape((dim,))
    return last_value  # return only the mean
