from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import hypoelliptic_diffusion_15_scheme

JAX_KEY = jax.random.PRNGKey(1337)

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


solver = ekf0


@partial(jnp.vectorize, signature="()->(c,s),(c,s,s)", excluded=(1,))
@partial(jax.jit, static_argnums=(1,))
def ibm(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_0000)

    x0 = jnp.ones((2,))
    init = x0
    if solver in [ekf0_2, ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, P0)

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T, N=1)

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
    return jnp.array([jnp.mean(sols[:, i], axis=0) for i in range(N + 1)]), jnp.array(
        [jnp.cov(sols[:, i], rowvar=False) for i in range(N + 1)])


@partial(jnp.vectorize, signature="()->(c,s),(c,s,s)", excluded=(1,))
@partial(jax.jit, static_argnums=(1,))
def ibm_15(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_0000)

    drift2 = lambda x: drift(x, 0)
    sigma2 = lambda x: sigma(x, 0)

    x0 = jnp.ones((2,))

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift2, sigma=sigma2, h=delta, N=N
        )

    linspaces, sols, *_ = wrapped_hypoelliptic_15(keys)

    return jnp.array([jnp.mean(sols[:, i], axis=0) for i in range(N + 1)]), jnp.array(
        [jnp.cov(sols[:, i], rowvar=False) for i in range(N + 1)])


deltas = jnp.array(
    [1 / 1000, 1 / 100, 1 / 10])

import pandas as pd

Ndeltas = jnp.ceil(1 / deltas)
res_mean = dict()
res_cov = dict()
res_cov_2 = dict()
for delta, n in zip(deltas, range(len(Ndeltas))):
    N = Ndeltas[n]
    mean, cov = ibm(delta, int(N))
    mean2, cov2 = ibm_15(delta, int(N))
    diff_mean = {k: float(jnp.linalg.norm((mean[k] - mean2[k]) / mean2[k])) for k in range(int(N) + 1)}
    diff_cov = {k: float(jnp.linalg.norm(cov[k] - cov2[k])) for k in range(int(N) + 1)}
    diff_cov_first_coordinate = {k: (cov[k][0, 0] - cov2[k][0, 0]) / cov2[k][0, 0] for k in range(int(N) + 1)}
    res_mean[float(delta)] = diff_mean
    res_cov[float(delta)] = diff_cov
    res_cov_2[float(delta)] = diff_cov_first_coordinate
df_mean = pd.DataFrame.from_dict(data=res_mean)
df_cov = pd.DataFrame.from_dict(data=res_cov)
df_cov_first_coordinate = pd.DataFrame.from_dict(data=res_cov_2)
df_mean.to_csv('res_mean.csv', header=False)
df_cov.to_csv('res_cov.csv', header=False)
df_cov_first_coordinate.to_csv('res_cov_first_coordinate.csv', header=False)

"""
IBM
M = jnp.array([[0.0, 1.0], [0.0, 0.0]])
C = jnp.array([[0.0], [1.0]])

def theoretical_mean(t):
    return jnp.array([1 + t, 1])


def theoretical_cov(t):
    return jnp.array([[1 / 3 * t ** 3, 1 / 2 * t ** 2], [1 / 2 * t ** 2, 1 * t]])


drift = lambda x, t: jnp.dot(M, x)
sigma = lambda x, t: C
"""
