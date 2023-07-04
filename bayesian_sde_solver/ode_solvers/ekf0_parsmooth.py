import jax
import jax.numpy as jnp
from parsmooth import filtering, FunctionalModel, MVNStandard
from parsmooth.linearization import extended

from bayesian_sde_solver.ode_solvers.probnum import interlace
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function


def solver(key, init, vector_field, h, N):
    #TODO: fix
    ts = jnp.linspace(0, N * h, N + 1)
    dim = init.shape[0]
    observations = jnp.zeros(N * dim).reshape((N, dim))

    def _observation_function(x, t):
        # IVP observation function
        return x[1, None] - vector_field(jax.lax.stop_gradient(x[0, None]), t) #does not work

    (
        _transition_function,
        _transition_mean,
        _transition_covariance,
    ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=dim)
    transition_model = FunctionalModel(
        _transition_function, MVNStandard(_transition_mean, _transition_covariance)
    )  #theta = 0, IBM

    # No initial variance
    init = MVNStandard(
        interlace(init, vector_field(init, 0.0)),
        jnp.zeros((2 * dim, 2 * dim))
    )

    # No noise
    observation_model = FunctionalModel(
        _observation_function, MVNStandard(jnp.zeros((dim,)), jnp.zeros((dim, dim)))
    )

    filtered = filtering(observations, init, transition_model, observation_model, extended, None,
                         params_transition=None, params_observation=(ts, ))
    last_value = jnp.vstack(filtered.mean[-1, ::2]).reshape((dim, ))

    return last_value
