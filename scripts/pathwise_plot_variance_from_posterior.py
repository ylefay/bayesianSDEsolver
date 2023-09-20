from functools import partial

import numpy.testing as npt

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function

from bayesian_sde_solver.utils.ivp import fhn

JAX_KEY = jax.random.PRNGKey(1337)

solver_name = "EKF0_2"
problem_name = "samples_from_prior_FHN"
prefix = f"{solver_name}_{problem_name}"
folder = "./"

_solver = ekf0_2

x0, drift, sigma = fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)

init = x0
if _solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, x0, P0)


def experiment(delta, N, M, fine, no_samples):
    prior = IOUP_transition_function(theta=0.0, sigma=1.0, dt=delta / M, q=1, dim=x0.shape[0])
    solver = partial(_solver, prior=prior, noise=None)

    def wrapped(_key, init, vector_field, T):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M)

    get_approx_fine = partial(_get_approx_fine, N=fine, dim=sigma(x0, 0.).shape[1])

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

    # stocking *coeffs can be memory expensive, see pathwise_convergence_experiment_2_RAM.py
    linspaces, sols, *coeffs = wrapped_filter_parabola(JAX_KEY)
    inc = coeffs[2]
    dt = delta / fine
    shape_incs = inc.shape
    assert fine == shape_incs[1]
    inc = inc.reshape((fine * shape_incs[0], shape_incs[2]))

    linspaces2, sols2 = euler_maruyama_pathwise(inc, init=x0, drift=drift, sigma=sigma, h=dt, N=fine * shape_incs[0])
    sampled_linspaces3, sols3 = euler_maruyama_pathwise(inc[::fine, ...], init=x0, drift=drift, sigma=sigma, h=delta,
                                                        N=shape_incs[0])  # EM
    sampled_linspaces2 = linspaces2[::fine, ...]  # fine EM
    sampled_sols2 = sols2[:, ::fine, ...]
    npt.assert_array_equal(sampled_linspaces2, sampled_linspaces3)
    _, mean_gaussian_pn, var_gaussian_pn = sols

    @partial(jnp.vectorize, signature='(n),(m),(m,m)->(m)')
    def sample(key_sample, mean, var):
        return mean + jlinalg.cholesky(var) @ jax.random.normal(key_sample, shape=mean_gaussian_pn.shape)

    KEY_SAMPLES = jax.random.split(JAX_KEY, no_samples)
    sampled_sols = sample(KEY_SAMPLES, mean_gaussian_pn, var_gaussian_pn)  # samples from PN
    return sampled_sols, sols2, sols3


deltas = 1 / jnp.array([16])
Ns = 1 / deltas
fineN = Ns ** 1.0
Mdeltas = jnp.ones((len(deltas),)) * (Ns) ** 0
T = 1.0
Ndeltas = T / deltas
no_samples = 100
for n in range(len(Ndeltas)):
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    fine = int(fineN[n])
    s1, s2, incs = experiment(delta, N, M, fine, no_samples)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
    jnp.save(f'{folder}/{prefix}_paths_{N}_{fine}', incs)
