from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import hypoelliptic_diffusion_15_scheme

JAX_KEY = jax.random.PRNGKey(1337)
solver = ekf0

gamma = 1.0
sig = 1.0
eps = 1.0
alpha = 1.0
s = 1.0

x0 = jnp.ones((2,))

init = x0
if solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, P0)


def drift(x, t):
    return jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
        [s / eps - x[0] ** 3 / eps, alpha])


def sigma(x, t):
    return jnp.array([[0.0], [sig]])


@partial(jnp.vectorize, signature="()->(d,n,s)", excluded=(1, 2,))
@partial(jax.jit, static_argnums=(1, 2,))
def scheme_ekf(delta, N=20, M=1):
    keys = jax.random.split(JAX_KEY, 1_000_00)

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

    _, sols, *_ = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2]:
        sols = sols[0]
    return sols


@partial(jnp.vectorize, signature="()->(d,n,s)", excluded=(1,))
@partial(jax.jit, static_argnums=(1,))
def scheme_ito_15(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_000_00)

    drift2 = lambda x: drift(x, 0)
    sigma2 = lambda x: sigma(x, 0)

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift2, sigma=sigma2, h=delta, N=N
        )

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    return sols


deltas = jnp.array(
    [1 / 2000, 1 / 1000, 1 / 100, 1 / 10])

Ndeltas = jnp.ceil(1 / deltas)
Mdeltas = jnp.ceil(1 / jnp.sqrt(deltas))
for delta, n in zip(deltas, range(len(Ndeltas))):
    N = Ndeltas[n]
    M = Mdeltas[n]
    sols = scheme_ekf(delta, int(N), int(M))
    sols2 = scheme_ito_15(delta, int(N))
    jnp.save(f'sols_{N}', sols)
    jnp.save(f'sols2_{N}', sols2)
