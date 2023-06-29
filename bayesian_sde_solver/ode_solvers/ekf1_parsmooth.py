import jax
from parsmooth import filtering, FunctionalModel, MVNStandard
from parsmooth.linearization import extended
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
import jax.numpy as jnp



def solver(key, init, vector_field, h, N):
    dim = init.shape[0]
    observations = jnp.zeros(N*dim).reshape((N, dim))
    _transition_function, _transition_mean, _transition_covariance = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h)
    transition_model = FunctionalModel(_transition_function, MVNStandard(_transition_mean, _transition_covariance))

    def _observation_function(x):
        return jnp.array([x[1] - vector_field(x[0], 0.0)])

    observation_model = FunctionalModel(_observation_function, MVNStandard(jnp.zeros((dim, )), jnp.zeros((dim, dim))))
    init = MVNStandard(jnp.array([init, vector_field(init, 0.0)]), jnp.zeros((dim, dim)))

    ts = jnp.linspace(0, N * h, N + 1)

    test_ = filtering(observations, init, transition_model, observation_model, extended)


    return test_
