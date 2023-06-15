import jax
import matplotlib.pyplot as plt
from bayesian_ode_solver.sde_solver import parabola_sde_solver_euler

drift = lambda x, t: x
sigma = lambda x, t: x
delta = 0.001
x0 = 1.0
N = 1000

JAX_KEY = jax.random.PRNGKey(1337)


@jax.vmap
def wrapped_parabola(key_op):
    return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)


keys = jax.random.split(JAX_KEY, 1_000)

linspaces, sols = wrapped_parabola(keys)
print(sols[:, -1].std())
print(sols[:, -1].mean())
