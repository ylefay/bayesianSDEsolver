from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise

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
    return 0 * x


def sigma(x, t):
    return jnp.array([[1.0]])


x0 = jnp.zeros((1,))
x0 = jnp.ones((2,))


def drift(x, t):
    return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x


def sigma(x, t):
    return jnp.array([[0.0], [1.0]])


drift_s, sigma_s = to_stratonovich(drift, sigma)

init = x0
if solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, P0)


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s),(d,k,l)", excluded=(1, 2, 3,))
@partial(jax.jit, static_argnums=(1, 2, 3,))
def experiment(delta, N, M, fine):
    keys = jax.random.split(JAX_KEY, 10_000)

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / M, N=M)

    get_approx_fine = partial(_get_approx_fine, N=fine)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift_s,
            sigma=sigma_s,
            x0=init,
            bm=get_approx_fine,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols, *coeffs = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2]:
        sols = sols[0]
    incs = coeffs[2]
    shape_incs = incs.shape
    assert fine == shape_incs[2]
    dt = 1 / (shape_incs[1] * shape_incs[2])
    incs = incs.reshape((shape_incs[0], fine * shape_incs[1], shape_incs[3]))
    incs *= jnp.sqrt(1 / dt)

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        return euler_maruyama_pathwise(inc, init=x0, drift=drift, sigma=sigma, h=dt, N=fine * shape_incs[1])

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    sampled_linspaces2 = linspaces2[:, ::fine, ...]
    sampled_sols2 = sols2[:, ::fine, ...]
    return sols, sampled_sols2, incs


deltas = jnp.array(
    [1 / 100, 1 / 90, 1 / 80, 1 / 70, 1 / 60, 1 / 50, 1 / 40, 1 / 30, 1 / 20, 1 / 10])  #
Ndeltas = jnp.ceil(1 / deltas)
Mdeltas = jnp.ones((len(deltas),))
Mdeltas = jnp.ceil(1 / jnp.sqrt(deltas))
Mdeltas = jnp.ones((len(deltas),)) * 1_000
fineDeltas = Mdeltas
folder = "./"
solver_name = "ekf0"
prefix = "IBM"
for delta, n in zip(deltas, range(len(Ndeltas))):
    N = Ndeltas[n]
    M = Mdeltas[n]
    fine = fineDeltas[n]
    s1, s2, incs = experiment(delta, int(N), int(M), int(fine))
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
    jnp.save(f'{folder}/{prefix}_paths_{N}_{fine}', incs)

