import jax
import jax.numpy as jnp

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.ode_solvers.probnum import ekf


def _solver(init, vector_field, h, N, sqrt=False, EKF0=False, prior=None, noise=None, n_states=2):
    """
    EKF{0, 1} implementation.
    IBM prior by default.
    One derivative of the vector field is used, q = 1.
    No observation noise by default, R = 0.

    Partially supports more than 2 states (i.e., more than y and the vector field)
    """
    ts = jnp.linspace(h, N * h, N)
    dim = int(init[0].shape[0] / n_states)
    if noise is None:
        noise = jnp.zeros((dim, dim))
    else:
        assert noise.shape == (dim, dim)

    if EKF0:
        def observation_function(x, t):
            # IVP observation function
            return x[1::n_states] - jax.lax.stop_gradient(vector_field(x[::n_states], t))
    else:
        def observation_function(x, t):
            # IVP observation function
            return x[1::n_states] - vector_field(x[::n_states], t)
    if prior is None:
        (
            transition_incr,
            transition_covariance,
            transition_matrix
        ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=n_states-1, dt=h, dim=dim)
    else:
        (
            transition_incr,
            transition_covariance,
            transition_matrix
        ) = prior
    if sqrt:
        transition_covariance = jnp.linalg.cholesky(transition_covariance)

    filtered = ekf(init=init, observation_function=observation_function, A=transition_matrix,
                   Q_or_cholQ=transition_covariance, xi=transition_incr, R_or_cholR=noise, params=(ts,), lower_sqrt=sqrt)

    return filtered
