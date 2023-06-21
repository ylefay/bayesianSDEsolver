import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.ode_solvers import ieks

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solver import sde_solver

def test_gbm_moment():
    # this tests the moments of the ieks method for the geometric brownian motion sde.

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1000)

    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.array([x])

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 100
    delta = 1 / N

    def wrapped_ieks(_key, init, vector_field, T):
        M = 100
        return ieks(key=None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_ieks_parabola(key_op):
        return sde_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                          ode_int=wrapped_ieks)

    linspaces, sols = wrapped_ieks_parabola(keys)

    npt.assert_almost_equal(sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1)
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)