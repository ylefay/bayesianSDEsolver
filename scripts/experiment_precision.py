import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver

solver = ekf0

JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)


def exp1(N, M):
    mu = 1.0
    sig = 1.0

    def drift(x, t):
        return mu * x

    def sigma(x, t):
        return jnp.array([[sig]])

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    init = x0
    if solver in [ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, P0)

    delta = 1 / N

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols = wrapped_filter_parabola(keys)
    if solver in [ekf1_2]:
        sols = sols[0]

    return sols