from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solvers import euler_maruyama
from bayesian_sde_solver.ssm_parabola import ekf0_marginal_parabola, ekf1_marginal_parabola
from bayesian_sde_solver.ssm_parabola import ssm_parabola_ode_solver

from bayesian_sde_solver.utils.ivp import square_matrix_fhn

JAX_KEY = jax.random.PRNGKey(1337)

solver_name = "EKF0_SSM"
problem_name = "FHN"
prefix = f"{solver_name}_{problem_name}"
folder = "./"

x0, drift, sigma, _, _ = square_matrix_fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)
init = x0

_solver = ekf0_marginal_parabola

@partial(jnp.vectorize, signature="()->(d,n,s)", excluded=(1, 2, 3))
def experiment(delta, N, M, fine):
    # Algorithm 4 implementation.
    # delta: mesh size of the Brownian approximations.
    # N: defines the total integration time: N*delta.
    # M: number of EKF pass. In the paper, we only consider M = 1.
    # fine: number of steps within an interval of length delta, for the fine Euler-Maruyama scheme.

    keys = jax.random.split(JAX_KEY, 100_000) #Number of samples.

    def solver(key, init, drift, diffusion, T):
        return _solver(key, init, delta=T, drift=drift, diffusion=diffusion, h=T / M, N=M, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op,
                                       drift=drift_s,
                                       sigma=sigma_s,
                                       x0=x0,
                                       delta=delta,
                                       N=N,
                                       solver=solver,
                                       )

    linspaces, sols = wrapped_filter_parabola(keys)

    @jax.vmap
    def wrapped_euler_maruyama(key_op):
        return euler_maruyama(key=key_op, init=x0, drift=drift, sigma=sigma, h=delta / fine, N=fine * N)

    # linspaces2, sols2 = wrapped_euler_maruyama(keys)
    # sampled_linspaces2 = linspaces2[:, ::fine, ...]
    # sampled_sols2 = sols2[:, ::fine, ...]

    # return sols, sampled_sols2
    return sols


deltas = 1 / jnp.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
Ns = 1 / deltas
fineN = Ns
Mdeltas = jnp.ones((len(deltas),)) * (Ns) ** 0.0
T = 1
Ndeltas = T / deltas

print(prefix)
for n in range(len(Ndeltas)):
    fine = int(fineN[n])
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    # s1, s2 = experiment(delta, N, M, fine)
    s1 = experiment(delta, N, M, fine)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    # jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
# warning: no path convergence, only moment.
