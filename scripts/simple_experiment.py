import jax

from bayesian_ode_solver.sde_solver import parabola_sde_solver_euler

drift = lambda x, t: 0
sigma = lambda x, t: 1
delta = 0.001
x0 = 1.0
N = 1

JAX_KEY = jax.random.PRNGKey(1337)


@jax.vmap
def wrapped_parabola(key_op):
    return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)


keys = jax.random.split(JAX_KEY, 1_000)

linspaces, sols = wrapped_parabola(keys)
# print(sol)
print(sols.mean())
print(sols.std())
# plt.plot(linspaces[0], sols.T)
