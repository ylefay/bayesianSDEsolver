from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.ode_solvers import euler
from bayesian_sde_solver.sde_solvers import euler_maruyama
from bayesian_sde_solver.utils.ivp import fhn

JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 2)

solver_name = "EKF1_2"
problem_name = "samples_from_prior_FHN"
prefix = f"{solver_name}_{problem_name}"
folder = "./"

_solver = ekf1_2

x0, drift, sigma, _, _ = fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)

init = x0
if _solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, x0, P0)


def experiment(delta, N, M, fine, no_samples):
    prior = IOUP_transition_function(theta=0., sigma=1.0, dt=delta / M, q=1, dim=x0.shape[0])
    solver = partial(_solver, prior=prior, noise=None, sqrt=True)
    get_approx_fine = partial(_get_approx_fine, N=fine, dim=sigma(x0, 0.).shape[1])
    dt = delta / fine

    def wrapped(_key, init, vector_field, T):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M)

    def wrapped_euler(_key, init, vector_field, T):
        _, y = euler_maruyama(key=_key, init=init, drift=vector_field,
                              sigma=lambda _, __: jnp.zeros((2, 1)), h=T / fine, N=fine)
        return y[-1]

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

    @jax.vmap
    def wrapped_filter_parabola_euler(key_op):
        return sde_solver(
            key=key_op,
            drift=drift_s,
            sigma=sigma_s,
            x0=x0,
            bm=get_approx_fine,
            delta=delta,
            N=N,
            ode_int=wrapped_euler,
        )

    # stocking *coeffs can be memory expensive, see pathwise_convergence_experiment_2_RAM.py
    linspaces, sols, *coeffs = wrapped_filter_parabola(keys)
    _, sols_parabola_ode_euler, *coeffs = wrapped_filter_parabola_euler(keys)
    inc = coeffs[2]
    shape_incs = inc.shape
    assert fine == shape_incs[2]
    total_inc = jnp.sum(inc, axis=2)
    inc = inc.reshape((shape_incs[0], fine * shape_incs[1], shape_incs[3]))

    linspaces2, sols2 = jax.vmap(
        lambda _inc: euler_maruyama_pathwise(_inc, init=x0, drift=drift, sigma=sigma, h=dt, N=fine * shape_incs[1]))(
        inc) # complete fine solution
    sampled_linspaces3, sols3 = jax.vmap(
        lambda _inc: euler_maruyama_pathwise(_inc, init=x0, drift=drift, sigma=sigma, h=delta,
                                             N=shape_incs[1]))(total_inc)  # approximate solution
    # subsampled fine solution
    sampled_linspaces2 = linspaces2[:, ::fine]
    sampled_sols2 = sols2[:, ::fine, ...]

    # We choose to keep only one simulated path
    mean_gaussian_pn, var_gaussian_pn = sols[1][0], sols[2][0]
    sols2 = sols2[0] # fine solution
    sols3 = sols3[0]
    sols_parabola_ode_euler = sols_parabola_ode_euler[0]

    mean_gaussian_pn = mean_gaussian_pn[1:] # get rid of the initializations t=0
    var_gaussian_pn = var_gaussian_pn[1:]

    n = mean_gaussian_pn.shape[0] # number of time points
    dim = mean_gaussian_pn.shape[1]
    mean = jnp.hstack(mean_gaussian_pn)
    var = jnp.block([[var_gaussian_pn[i] if i == j else jnp.zeros((dim, dim)) for i in range(n)] for j in range(n)])

    @partial(jnp.vectorize, signature='(n)->(m)')
    def sample(key_sample):
        return mean + jlinalg.cholesky(var) @ jax.random.normal(key_sample, shape=mean.shape)

    KEY_SAMPLES = jax.random.split(JAX_KEY, no_samples)
    sampled_sols = sample(KEY_SAMPLES) # samples from PN posteriors
    sampled_sols = sampled_sols.reshape((no_samples, n, dim)) # reshapping into (no_samples, n, dim) after collapsing the shape to ease the sampling procedure.
    return sampled_sols, mean_gaussian_pn, var_gaussian_pn, sols2, sols3, sols_parabola_ode_euler, inc, coeffs[0], \
           coeffs[1]


deltas = 1 / jnp.array([9])
Ns = 1 / deltas
fineN = Ns ** 1.0
Mdeltas = jnp.ones((len(deltas),)) * (Ns) ** 0
T = 1
Ndeltas = T / deltas
no_samples = 100
for n in range(len(Ndeltas)):
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    fine = int(fineN[n])
    with jax.disable_jit(False):
        sampled_sols, mean, var, s_fine, s_em, s_parabola_ode, fine_incs, \
        incs, parabola_coeffs = experiment(delta, N, M, fine, no_samples)
    jnp.save(f'{folder}/{prefix}_sampled_sols_{N}_{M}', sampled_sols)
    jnp.save(f'{folder}/{prefix}_mean_pn_{N}_{M}', mean)
    jnp.save(f'{folder}/{prefix}_var_pn_{N}_{M}', var)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s_fine)
    jnp.save(f'{folder}/{prefix}_pathwise_sols3_{N}_{fine}', s_em)
    jnp.save(f'{folder}/{prefix}_pathwise_sols4_parabola_ode_{N}_{fine}', s_parabola_ode)

    jnp.save(f'{folder}/{prefix}_fine_incs_{N}_{fine}', fine_incs)
    jnp.save(f'{folder}/{prefix}_incs_{N}_{fine}', incs)
    jnp.save(f'{folder}/{prefix}_parabola_coeffs_{N}_{fine}', parabola_coeffs)
