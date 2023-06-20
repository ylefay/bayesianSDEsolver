import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.sde_solver import sde_solver


def harmonic_oscillator_probdiff():
    #same as the test function in test/test_sde_solver_foster.py
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 10_00)

    gamma = 1.0
    D = 1.0
    sig = 2.0

    drift = lambda x, t: jnp.dot(jnp.array([[0.0, 1.0], [-D, -gamma]]), x)
    sigma = lambda x, t: jnp.array([[0.0], [sig]])

    x0 = jnp.ones((2,))
    N = 100
    delta = 2 / N

    def wrapped_ekf0(_key, init, vector_field, T):
        N = 10
        return ekf0(init=init, vector_field=vector_field, h=T / N, N=N)
    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, bm=lambda: parabola_approx(1), delta=delta, N=N,
                          ode_int=wrapped_ekf0)

    linspaces, sols = wrapped_filter_parabola(keys)
