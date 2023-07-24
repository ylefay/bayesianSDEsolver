from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import hypoelliptic_diffusion_15_scheme

JAX_KEY = jax.random.PRNGKey(1337)

M = jnp.array([[0.0, 1.0], [0.0, 0.0]])
C = jnp.array([[0.0], [1.0]])


@partial(jnp.vectorize, signature="()->(c,s),(c,s,s)", excluded=(1,))
@partial(jax.jit, static_argnums=(1,))
def ibm(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_000_000)

    drift = lambda x, t: jnp.dot(M, x)
    sigma = lambda x, t: C

    x0 = jnp.ones((2,))

    def wrapped(_key, init, vector_field, T):
        return ekf0(key=None, init=init, vector_field=vector_field, h=T / 1, N=1)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    _, sols, *_ = wrapped_filter_parabola(keys)
    return jnp.array([jnp.mean(sols[:, i], axis=0) for i in range(N + 1)]), jnp.array(
        [jnp.cov(sols[:, i], rowvar=False) for i in range(N + 1)])


@partial(jnp.vectorize, signature="()->(c,s,s)", excluded=(1,))
@partial(jax.jit, static_argnums=(1,))
def ibm_15(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_000_000)

    drift = lambda x: jnp.dot(M, x)
    sigma = lambda x: C

    x0 = jnp.ones((2,))

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift, sigma=sigma, h=delta, N=N
        )

    linspaces, sols, *_ = wrapped_hypoelliptic_15(keys)

    return jnp.cov(sols, rowvar=False)


deltas = jnp.array(
    [ 1 / 200, 1 / 100, 1 / 50, 1 / 20, 1 / 10, 1 / 5, 1 / 2, 1])


def theoretical_mean(t):
    return jnp.array([1 + t, 1])


def theoretical_cov(t):
    return jnp.array([[1 / 3 * t ** 3, 1 / 2 * t ** 2], [1 / 2 * t ** 2, 1 * t]])


import pandas as pd

Ndeltas = jnp.ceil(1 / deltas)
res_mean = dict()
res_cov = dict()
for delta, n in zip(deltas, range(len(Ndeltas))):
    N = Ndeltas[n]
    mean, cov = ibm(delta, int(N))

    diff_mean = {k: float(jnp.linalg.norm(mean[k] - theoretical_mean(delta * k))) for k in range(int(N) + 1)}
    diff_cov = {k: float(jnp.linalg.norm(cov[k] - theoretical_cov(delta * k))) for k in range(int(N) + 1)}

    res_mean[float(delta)] = diff_mean
    res_cov[float(delta)] = diff_cov
df_mean = pd.DataFrame.from_dict(data=res_mean)
df_cov = pd.DataFrame.from_dict(data=res_cov)
df_mean.to_csv('res_mean2.csv', header=False)
df_cov.to_csv('res_cov2.csv', header=False)
