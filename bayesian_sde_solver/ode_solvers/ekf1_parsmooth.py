import jax
import jax.numpy as jnp
from parsmooth import filtering, FunctionalModel, MVNStandard
from parsmooth.linearization import extended

from bayesian_sde_solver.ode_solvers.probnum import interlace
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function


def solver(key, init, vector_field, h, N):
    dim = init.shape[0]
    observations = jnp.zeros(N * dim).reshape((N, dim))

    (
        _transition_function,
        _transition_mean,
        _transition_covariance,
    ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=dim)
    transition_model = FunctionalModel(
        _transition_function, MVNStandard(_transition_mean, _transition_covariance)
    )  #theta = 0, IBM

    def _observation_function(x, t):
        # IVP observation function
        return x[1, None] - vector_field(x[0, None], t)

    # No initial variance
    init = MVNStandard(
        interlace(init, vector_field(init, 0.0)),
        jnp.zeros((2 * dim, 2 * dim))
    )

    # No noise
    observation_model = FunctionalModel(
        _observation_function, MVNStandard(jnp.zeros((dim,)), jnp.zeros((dim, dim)))
    )

    ts = jnp.linspace(0, N * h, N + 1)

    filtered = filtering(observations, init, transition_model, observation_model, extended, None,
                         params_transition=None, params_observation=(ts, ))
    last_value = jnp.vstack(filtered.mean[-1, ::2]).reshape((dim, ))

    if key is not None:
        last_sample = filtered.mean[-1] + jnp.linalg.cholesky(filtered.cov[-1]) @ \
            jax.random.multivariate_normal(key, jnp.zeros((2 * dim, )), jnp.eye(2 * dim))
        return jnp.vstack(last_sample[::2]).reshape((dim, ))  #nan

    return last_value
