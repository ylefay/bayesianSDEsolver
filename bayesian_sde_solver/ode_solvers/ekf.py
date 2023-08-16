import jax
import jax.numpy as jnp

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.ode_solvers.probnum import ekf


def _solver(init, vector_field, h, N, sqrt=False, EKF0=False):
    """
    EKF{0, 1} implementation.
    IOUP prior.
    One derivative of the vector field is used, q = 1.
    No observation noise, R = 0.
    """
    ts = jnp.linspace(h, N * h, N)
    dim = int(init[0].shape[0] / 2)
    noise = jnp.zeros((dim, dim))

    if EKF0:
        def observation_function(x, t):
            # IVP observation function
            return x[1::2] - jax.lax.stop_gradient(vector_field(x[::2], t))
    else:
        def observation_function(x, t):
            # IVP observation function
            return x[1::2] - vector_field(x[::2], t)

    (
        _,
        transition_covariance,
        transition_matrix
    ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=dim)

    if sqrt:
        transition_covariance = jnp.linalg.cholesky(transition_covariance)

    filtered = ekf(init=init, observation_function=observation_function, A=transition_matrix,
                   Q_or_cholQ=transition_covariance, R_or_cholR=noise, params=(ts,), lower_sqrt=sqrt)

    return filtered
