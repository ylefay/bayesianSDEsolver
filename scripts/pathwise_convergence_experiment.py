from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_and_brownian as parabola_approx_and_brownian
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_piecewise

JAX_KEY = jax.random.PRNGKey(1337)
solver = ekf0

gamma = 1.0
sig = 1.0
eps = 1.0
alpha = 1.0
s = 1.0


def drift(x, t):
    return jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
        [s / eps - x[0] ** 3 / eps, alpha])


def sigma(x, t):
    return jnp.array([[0.0], [sig]])


def drift(x, t):
    return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x


def sigma(x, t):
    return jnp.array([[0.0], [1.0]])


x0 = jnp.ones((2,))


# GBM:
def drift(x, t):
    return x


def sigma(x, t):
    return jnp.array([[1.0]]) @ x


x0 = jnp.ones((1,))

init = x0
if solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, P0)


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s)", excluded=(1, 2,))
@partial(jax.jit, static_argnums=(1, 2,))
def experiment(delta, N, M):
    keys = jax.random.split(JAX_KEY, 1_000_0)

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

    linspaces, sols, *coeffs, incs = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2]:
        sols = sols[0]

    shape_incs = incs.shape
    incs = incs.reshape((shape_incs[0], shape_incs[1] * shape_incs[2], shape_incs[3]))

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        P = shape_incs[1] * shape_incs[2]
        h = 1 / P
        return euler_maruyama_piecewise(inc, init=x0, drift=drift, sigma=sigma, h=h, N=P)

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    sampled_linspaces2 = linspaces2[:, ::shape_incs[2], ...]
    sampled_sols2 = sols2[:, ::shape_incs[2], ...]
    return sols, sampled_sols2


deltas = jnp.array(
    [1 / 4000, 1 / 3000, 1 / 2000, 1 / 1000, 1 / 100])  #
Mdeltas = jnp.ones((len(deltas),))
Ndeltas = jnp.ceil(1 / deltas)
Mdeltas = jnp.ceil(1 / jnp.sqrt(deltas))
prefix = "GBM"
for delta, n in zip(deltas, range(len(Ndeltas))):
    N = Ndeltas[n]
    M = Mdeltas[n]
    s1, s2 = experiment(delta, int(N), int(M))
    jnp.save(f'{prefix}_pathwise_sols_{N}', s1)
    jnp.save(f'{prefix}_pathwise_sols2_{N}', s2)
