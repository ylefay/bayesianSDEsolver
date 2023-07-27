import jax
import jax.numpy as jnp
import pandas as pd

from bayesian_sde_solver.foster_polynomial import get_approx_and_brownian as parabola_approx_and_brownian
from bayesian_sde_solver.ode_solvers import ekf0_2
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_piecewise

JAX_KEY = jax.random.PRNGKey(1337)

M = jnp.array([[0.0, 1.0], [0.0, 0.0]])
C = jnp.array([[0.0], [1.0]])

solver = ekf0_2
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1)
drift = lambda x, t: jnp.dot(M, x)
sigma = lambda x, t: C

x0 = jnp.ones((2,))
init = x0
P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
init = (x0, P0)
T = 1


def experiment(N):
    delta = T / N

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / 1, N=1)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=parabola_approx_and_brownian,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols, *coeffs, incs = wrapped_filter_parabola(keys)
    shape_incs = incs.shape
    incs = incs.reshape((shape_incs[0], shape_incs[1] * shape_incs[2], shape_incs[3]))

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        P = shape_incs[1] * shape_incs[2]
        h = T / P
        return euler_maruyama_piecewise(inc, init=x0, drift=drift, sigma=sigma, h=h, N=P)

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    sampled_linspaces2 = linspaces2[:, ::shape_incs[2], ...]
    sampled_sols2 = sols2[:, ::shape_incs[2], ...]
    return linspaces, sols, sampled_linspaces2, sampled_sols2


N = 1000

linspaces, s1, _, s2 = experiment(N)
y_m, y_cov = s1[0][0, :, 0], s1[1][0, :, 0, 0]
y = s2[0, :, 0]

res = pd.DataFrame.from_dict({'index': linspaces[0], 'Esp': y_m, 'Cov': y_cov, 'Euler': y})
res.to_csv('ekf0_2.csv')
