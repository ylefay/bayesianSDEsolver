import jax
from bayesian_ode_solver.ode_solvers import euler
from bayesian_ode_solver.sde_solver import sde_solver
from bayesian_ode_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_ode_solver.ito_stratonovich import to_stratonovich
import numpy.testing as npt
import jax.numpy as jnp



def test_gbm():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * x

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = 1.0
    N = 100
    delta = 1 / N

    def wrapped_euler(_key, init, vector_field, T):
        # 10 points euler
        M = 100
        return euler(init=init, vector_field=vector_field, h=T / M, N=M)

    def parabola_sde_solver_euler(key, drift, sigma, x0, delta, N):
        return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                          ode_int=wrapped_euler)

    @jax.vmap
    def wrapped_parabola(key_op):
        return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)

    linspaces, sols = wrapped_parabola(keys)
    npt.assert_almost_equal(sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1)
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)
