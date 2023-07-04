import jax
import jax.numpy as jnp
import numpy.testing as npt

from parsmooth import MVNStandard

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.ode_solvers import ekf1_2

def test_gbm_ekf1():
    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    m0 = jnp.ones((1, ))
    P0 = jnp.zeros((1, 1))
    x0 = MVNStandard(m0, P0)

    N = 1
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    def wrapped_ekf1(_key, init, vector_field, T):
        M = 200
        return ekf1_2(key=None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped_ekf1,
        )
    linspaces, sols = wrapped_filter_parabola(keys)
    trajectory, P = sols
    print(trajectory[:, -1].std())
    print(trajectory[:, -1].mean())
    npt.assert_almost_equal(
        trajectory[:, -1].std(), m0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(trajectory[:, -1].mean(), m0 * jnp.exp(a), decimal=1)
