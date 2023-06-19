import jax
import jax.numpy as jnp
from probdiffeq import ivpsolvers, ivpsolve
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solver import sde_solver

drift = lambda x, t: x
sigma = lambda x, t: jnp.diag(x)

drift, sigma = to_stratonovich(drift, sigma)

x0 = jnp.ones((1,))
N = 100
delta = 1 / N

JAX_KEY = jax.random.PRNGKey(1337)


def wrapped_diffeq_filter(_key, init, vector_field, T):
    N = 10

    def vf(x, t=0.0, p=None):
        return vector_field(x, t)

    ts0 = ivpsolvers.solver_calibrationfree(*filters.filter(*recipes.ts0_iso(ode_order=1, num_derivatives=1)))
    ts = jnp.linspace(0, T, N + 1)
    solution = ivpsolve.solve_and_save_at(vector_field=vf, initial_values=(init,), solver=ts0, save_at=ts, dt0=T / N)
    return solution.marginals.mean[-1, 0,
           :]  # return only the mean of the last point of the trajectory, you may want the covariance as well


def parabola_sde_solver_filter(key, drift, sigma, x0, delta, N):
    return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                      ode_int=wrapped_diffeq_filter)


@jax.vmap
def wrapped_filter_parabola(key_op):
    return parabola_sde_solver_filter(key_op, drift, sigma, x0, delta, N)


keys = jax.random.split(JAX_KEY, 1_000)

linspaces, sols = wrapped_filter_parabola(keys)
print(sols[:, -1].mean())
