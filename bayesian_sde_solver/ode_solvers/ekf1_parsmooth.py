import jax
import jax.numpy as jnp
from parsmooth import filtering, FunctionalModel, MVNStandard
from parsmooth.linearization import extended

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function


def interlace(x, y):
    return jnp.vstack((x, y)).reshape((-1,), order='F')


def solver(key, init, vector_field, h, N):
    dim = init.shape[0]
    observations = jnp.zeros((N - 1) * dim).reshape((N - 1, dim))

    (
        _transition_function,
        _transition_mean,
        _transition_covariance,
    ) = IOUP_transition_function(theta=0., sigma=1.0, q=1, dt=h, dim=dim)
    transition_model = FunctionalModel(
        _transition_function, MVNStandard(_transition_mean, _transition_covariance)
    )

    def _observation_function(x):
        return x[0, None] - vector_field(x[1, None], 1.0)

    init = MVNStandard(
        interlace(init, vector_field(init, 0.0)),
        jnp.zeros((2 * dim, 2 * dim))
    )

    observation_model = FunctionalModel(
        _observation_function, MVNStandard(jnp.zeros((dim,)), jnp.zeros((dim, dim)))
    )

    ts = jnp.linspace(0, N * h, N + 1)

    filtered = filtering(observations, init, transition_model, observation_model, extended)
    last_value = jnp.vstack(filtered.mean[-1, ::2]).reshape((dim, ))

    if key is not None:
        last_sample = filtered.mean[-1] + jnp.linalg.cholesky(filtered.cov[-1]) @ \
            jax.random.multivariate_normal(key, jnp.zeros((2 * dim, )), jnp.eye(2 * dim))
        return jnp.vstack(last_sample[::2]).reshape((dim, ))

    return last_value
