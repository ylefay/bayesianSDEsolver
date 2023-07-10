from functools import partial

import jax
import jax.numpy as jnp
from bayesian_sde_solver.foster_polynomial import get_approx as _parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver

JAX_KEY = jax.random.PRNGKey(1337)

M, N = 6, 100
h = 1.0
K = 1000
bm_key, exp_key = jax.random.split(JAX_KEY, 2)
exp_keys = jax.random.split(exp_key, M * N)
bms = jax.random.normal(bm_key, shape=(M * K - 1, 1, 1))
bms = jnp.insert(bms, 0, 0, axis=0)
bms = bms * h ** 0.5 / K ** 0.5
bms = jnp.cumsum(bms, axis=0)
bms_inc = bms[::K - 1]

solver = ekf0


@jax.jit
def get_approx(bms_inc, dim=1):
    _, eval_approx = _parabola_approx()

    def parabola_approx(key, dt):
        assert len(key) == len(bms_inc)
        eps_1 = jax.random.normal(key, shape=(1, dim))
        eps_1 *= jnp.sqrt(0.5 * dt)
        return bms_inc, eps_1

    return parabola_approx, eval_approx


mu = 1.0
sig = 1.0


def drift(x, t):
    return mu * x


def sigma(x, t):
    return jnp.array([[sig]])  # jnp.diag(x) does not work, multiplicative noise?


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
        bm=partial(get_approx, bms_inc=bms_inc),
        delta=delta,
        N=N,
        ode_int=wrapped,
    )


N = 1000
M = 100
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)
linspaces, sols = wrapped_filter_parabola(keys)
