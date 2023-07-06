import jax
import jax.numpy as jnp

from bayesian_sde_solver.ode_solvers.probnum.source import ekf
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.ode_solvers.probnum import interlace


def _solver(init, vector_field, h, N, sqrt=False):
    """
    EKF1 implementation.
    Taking init, defining the prior.
    One derivative of the vector field is used.
    No observation noise.
    """
    ts = jnp.linspace(0, N * h, N + 1)
    dim = int(init[0].shape[0] / 2)

    def _observation_function(x, t):
        # IVP observation function
        return x[1::2] - vector_field(x[::2], t)

    (
        _,
        _transition_mean,
        _transition_covariance,
        _transition_matrix
    ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=dim)

    if sqrt:
        _transition_covariance = jnp.linalg.cholesky(_transition_covariance)

    filtered = ekf(init, _observation_function, _transition_matrix, _transition_covariance, jnp.zeros((dim, dim)), (ts,), sqrt)

    return filtered[-1]

def solver(key, init, vector_field, h, N):
    """
    Wrapper for EKF1 with new prior at each step.
    """
    dim = init.shape[0]
    # Zero initial variance
    init = (
        interlace(init, vector_field(init, 0.0)),
        jnp.zeros((2 * dim, 2 * dim))
    )
    filtered = _solver(init, vector_field, h, N, sqrt=True)
    m, chol = filtered
    if key is not None:
        last_sample = m + chol @ \
                      jax.random.multivariate_normal(key, jnp.zeros((2 * dim,)), jnp.eye(2 * dim))
        return jnp.vstack(last_sample[::2]).reshape((dim,))
    last_value = jnp.vstack(m[::2]).reshape((dim,))
    return last_value  # return only the mean
