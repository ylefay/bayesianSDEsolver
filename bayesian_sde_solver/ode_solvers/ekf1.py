import jax
import jax.numpy as jnp
from parsmooth import filtering, FunctionalModel, MVNSqrt, MVNStandard
from parsmooth.linearization import extended

from bayesian_sde_solver.ode_solvers.probnum import interlace
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function


def _solver(init, vector_field, h, N):
    """
    EKF1 implementation using parsmooth.
    Taking init as a MVNSqrt or MVNStandard object, defining the prior.
    One derivative of the vector field is used.
    No observation noise.
    """
    ts = jnp.linspace(0, N * h, N + 1)
    dim = int(init.mean.shape[0] / 2)
    observations = jnp.zeros(N * dim).reshape((N, dim))

    def _observation_function(x, t):
        # IVP observation function
        # TODO: does not work in the multivariate case.
        return x[1, None] - vector_field(x[0, None], t)

    (
        _transition_function,
        _transition_mean,
        _transition_covariance,
    ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=dim)

    if isinstance(init, MVNSqrt):
        transition_model = FunctionalModel(
            _transition_function, MVNSqrt(_transition_mean, jnp.linalg.cholesky(_transition_covariance))
        )  #theta = 0, IBM, num_derivative = 1
        # No noise
        observation_model = FunctionalModel(
            _observation_function, MVNSqrt(jnp.zeros((dim,)), jnp.zeros((dim, dim)))
        )
    else:
        transition_model = FunctionalModel(
            _transition_function, MVNStandard(_transition_mean, _transition_covariance)
        )
        # No noise
        observation_model = FunctionalModel(
            _observation_function, MVNStandard(jnp.zeros((dim,)), jnp.zeros((dim, dim)))
        )



    filtered = filtering(observations, init, transition_model, observation_model, extended, None,
                         params_transition=None, params_observation=(ts, ))

    if isinstance(init, MVNSqrt):
        return MVNSqrt(filtered.mean[-1], filtered.chol[-1])
    else:
        return MVNStandard(filtered.mean[-1], filtered.cov[-1])

def solver(key, init, vector_field, h, N):
    """
    Wrapper for EKF1 with new prior at each step.
    """
    dim = init.shape[0]
    # No initial variance
    init = MVNSqrt(
        interlace(init, vector_field(init, 0.0)),
        jnp.zeros((2 * dim, 2 * dim))
    )
    filtered = _solver(init, vector_field, h, N)
    m, chol = filtered
    if key is not None:
        last_sample = m + chol @ \
            jax.random.multivariate_normal(key, jnp.zeros((2 * dim, )), jnp.eye(2 * dim))
        return jnp.vstack(last_sample[::2]).reshape((dim, ))
    last_value = jnp.vstack(m[::2]).reshape((dim, ))
    return last_value #return only the mean
