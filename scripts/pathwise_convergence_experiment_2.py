from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
JAX_KEY = jax.random.PRNGKey(1337)
solver = ekf0

gamma = 1.0
sig = 1.0
eps = 1.0
alpha = 1.0
s = 1.0

x0 = jnp.ones((2,))


_solver = ekf0
theta = 1.0

def drift(x, t):
    return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x


def sigma(x, t):
    return jnp.array([[0.0], [1.0]])


def drift(x, t):
    return jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
        [s / eps - x[0] ** 3 / eps, alpha])


def sigma(x, t):
    return jnp.array([[0.0], [sig]])


def drift(x, t):
    def pi(x):
        return jnp.exp(-x @ x.T / 2)

    return jax.jacfwd(lambda z: jnp.log(pi(z)))(x)


def sigma(x, t):
    return jnp.array([[jnp.sqrt(2)]])


x0 = jnp.ones((1,))

drift_s, sigma_s = to_stratonovich(drift, sigma)

init = x0
if solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, P0)


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s),(d,k,l)", excluded=(1, 2, 3,))
@partial(jax.jit, static_argnums=(1, 2, 3,))
def experiment(delta, N, M, fine):
    keys = jax.random.split(JAX_KEY, 1_0000)

    prior = IOUP_transition_function(theta=0.0, sigma=1.0, dt=delta / M, q=1, dim=x0.shape[0])
    solver = partial(_solver, prior=prior, noise=jnp.array([[0.0]]))

    def wrapped(_key, init, vector_field, T):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M)

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

    #stocking *coeffs can be memory expensive, see pathwise_convergence_experiment_2_RAM.py
    linspaces, sols, *coeffs = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2]:
        sols = sols[0]
    incs = coeffs[2]
    dt = delta / fine
    shape_incs = incs.shape
    assert fine == shape_incs[2]
    incs = incs.reshape((shape_incs[0], fine * shape_incs[1], shape_incs[3]))
    incs *= jnp.sqrt(1 / dt)

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        return euler_maruyama_pathwise(inc, init=x0, drift=drift, sigma=sigma, h=dt, N=fine * shape_incs[1])

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    sampled_linspaces2 = linspaces2[:, ::fine, ...]
    sampled_sols2 = sols2[:, ::fine, ...]
    return sols, sampled_sols2, incs


Ns = jnp.array([4, 8, 16, 32, 64, 128])
deltas = jnp.array([0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
fineDeltas = Ns ** 1
Mdeltas = jnp.ones((len(deltas),)) * Ns ** 0
Ndeltas = Ns

folder = "./"
solver_name = "ekf0"
problem_name = "langevin"
prefix = f"{solver_name}_{problem_name}"
for n in range(len(Ndeltas)):
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    fine = int(fineDeltas[n])
    s1, s2, incs = experiment(delta, N, M, fine)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
    jnp.save(f'{folder}/{prefix}_paths_{N}_{fine}', incs)
