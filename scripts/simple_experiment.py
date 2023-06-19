import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import euler
from bayesian_sde_solver.sde_solver import sde_solver

drift = lambda x, t: x
sigma = lambda x, t: jnp.array([x])

drift, sigma = to_stratonovich(drift, sigma)

x0 = jnp.ones((1,))
N = 100
delta = 1 / N

JAX_KEY = jax.random.PRNGKey(1337)

M = 100  # points for the Euler method


def wrapped_euler(_key, init, vector_field, T):
    return euler(init=init, vector_field=vector_field, h=T / M, N=M)


def parabola_sde_solver_euler(key, drift, sigma, x0, delta, N):
    return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                      ode_int=wrapped_euler)


@jax.vmap
def wrapped_parabola(key_op):
    return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)


keys = jax.random.split(JAX_KEY, 1_000)

linspaces, sols = wrapped_parabola(keys)
print(sols[:, -1].mean())
