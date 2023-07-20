import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from bayesian_sde_solver.foster_polynomial import get_approx_and_brownian as parabola_approx_and_brownian
from bayesian_sde_solver.ode_solvers import ekf0 as solver
from bayesian_sde_solver.sde_solver import sde_solver

N = 10
M = 5
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_00)

gamma = 1.0
D = 1.0
sig = 2.0

Mm = jnp.array([[0.0, 1.0], [-D, -gamma]])
C = jnp.array([[0.0], [sig]])


def drift(x, t):
    return jnp.dot(Mm, x)


def sigma(x, t):
    return C


x0 = jnp.ones((2,))
init = x0
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
        bm=parabola_approx_and_brownian,
        delta=delta,
        N=N,
        ode_int=wrapped,
    )


with jax.disable_jit():
    linspaces, sols, *coeffs, paths = wrapped_filter_parabola(keys)

incs = paths[..., -1, :]
_incs = jnp.roll(paths[..., -1, :], 1, axis=1)
_incs = _incs.at[:, 0, ...].set(0)
_incs = _incs[..., jnp.newaxis, :]
_incs = jnp.repeat(_incs, paths.shape[2], axis=2)
paths2 = paths + _incs
pass