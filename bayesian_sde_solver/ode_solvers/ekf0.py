import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from bayesian_sde_solver.ode_solvers.ekf1 import _solver
from bayesian_sde_solver.ode_solvers.probnum import interlace


def solver(key, init, vector_field, h, N, sqrt=False, prior=None, noise=None):
    """
    Wrapper for EKF0 with new prior at each step.
    Algorithm 2.
    """
    n_states = int(prior[2].shape[0] / init.shape[0])
    dim = init.shape[0]
    # the first two cases are exact initialization
    if n_states == 2:
        # Zero initial variance
        init = (
            interlace((init, vector_field(init, 0.0))),
            jnp.zeros((n_states * dim, n_states * dim))
        )
    elif n_states == 3:
        # Zero initial variance
        init = (
            interlace((init, vector_field(init, 0.0), jax.jacfwd(vector_field)(init, 0.0) @ vector_field(init, 0.0))),
            jnp.zeros((n_states * dim, n_states * dim))
        )
    else:
        # The three first states are exactly initialized, otherwise set to 0 with unit variance. This is arbitrary.
        init = (
            interlace((init, vector_field(init, 0.0),
                       jax.jacfwd(vector_field, argnums=(0,))(init, 0.0) @ vector_field(init, 0.0),
                       *(jnp.zeros(dim) for _ in range(n_states - 3)))),
            jnp.eye(n_states * dim).at[:3 * dim, :3 * dim].set(0.0)
        )
    filtered = _solver(init, vector_field, h, N, sqrt, EKF0=True, prior=prior, noise=noise, n_states=n_states)
    m, P = filtered
    if key is not None:
        sqrtP = P if sqrt else jnp.real(linalg.sqrtm(P))
        last_sample = m + sqrtP @ jax.random.multivariate_normal(key, jnp.zeros((n_states * dim,)),
                                                                 jnp.eye(n_states * dim))
        return jnp.vstack(last_sample[::n_states]).reshape((dim,))
    last_value = jnp.vstack(m[::n_states]).reshape((dim,))
    return last_value  # return only the mean
