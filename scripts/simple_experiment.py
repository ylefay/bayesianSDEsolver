import jax
import matplotlib.pyplot as plt
from bayesian_ode_solver.ode_solvers import euler
from bayesian_ode_solver.sde_solver import sde_solver
from bayesian_ode_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_ode_solver.ito_stratonovich import to_stratonovich

drift = lambda x, t: x
sigma = lambda x, t: x

drift, sigma = to_stratonovich(drift, sigma)

x0 = 1.0
N = 100
delta = 1 / N


JAX_KEY = jax.random.PRNGKey(1337)

def wrapped_euler(_key, init, vector_field, T):
    # 10 points euler
    M = 100
    return euler(init=init, vector_field=vector_field, h=T / M, N=M)


def parabola_sde_solver_euler(key, drift, sigma, x0, delta, N):
    return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N, ode_int=wrapped_euler)

@jax.vmap
def wrapped_parabola(key_op):
    return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)


keys = jax.random.split(JAX_KEY, 1_000)

linspaces, sols = wrapped_parabola(keys)
print(sols[:, -1].std())
print(sols[:, -1].mean())
