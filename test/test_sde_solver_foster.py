import jax
from bayesian_sde_solver.ode_solvers import euler
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
import numpy.testing as npt
import jax.numpy as jnp
from probdiffeq import ivpsolvers, ivpsolve
from probdiffeq.strategies import filters
from probdiffeq.statespace import recipes


def test_gbm_euler():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.array([x])

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
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

def test_gbm_probdiff_eq():

    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 100
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    def wrapped_diffeq_filter(_key, init, vector_field, T):
        N = 10

        def vf(x, t=0.0, p=None):
            return vector_field(x, t)

        ts0 = ivpsolvers.solver_calibrationfree(*filters.filter(*recipes.ts0_iso(ode_order=1, num_derivatives=1)))
        ts = jnp.linspace(0, T, N + 1)
        solution = ivpsolve.solve_and_save_at(vector_field=vf, initial_values=(init,), solver=ts0, save_at=ts,
                                              dt0=T / N)
        return solution.marginals.mean[-1, 0,
               :]  # return only the mean of the last point of the trajectory, you may want the covariance as well

    def parabola_sde_solver_filter(key, drift, sigma, x0, delta, N):
        return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                          ode_int=wrapped_diffeq_filter)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return parabola_sde_solver_filter(key_op, drift, sigma, x0, delta, N)


    linspaces, sols = wrapped_filter_parabola(keys)
    npt.assert_almost_equal(sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1)
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)