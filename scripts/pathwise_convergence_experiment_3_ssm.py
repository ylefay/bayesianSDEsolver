from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solvers import euler_maruyama
from bayesian_sde_solver.ssm_parabola import ekf0_marginal_parabola
from bayesian_sde_solver.ssm_parabola import ssm_parabola_ode_solver

import ivp
JAX_KEY = jax.random.PRNGKey(1337)



x0, drift, sigma = ivp.square_matrix_fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)
init = x0


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s)", excluded=(1, 2, 3))
def experiment(delta, N, M, fine):
    keys = jax.random.split(JAX_KEY, 1_000)

    def solver(key, init, delta, drift, diffusion, T):
        return ekf0_marginal_parabola(key, init, delta, drift, diffusion, h=T / M, N=M, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op,
                                       drift=drift,
                                       sigma=sigma,
                                       x0=x0,
                                       delta=delta,
                                       N=N,
                                       solver=solver,
                                       )

    linspaces, sols, *coeffs = wrapped_filter_parabola(keys)

    @jax.vmap
    def wrapped_euler_maruyama(key_op):
       return euler_maruyama(key=key_op, init=x0, drift=drift, sigma=sigma, h=delta/fine, N=fine * N)

    linspaces2, sols2 = wrapped_euler_maruyama(keys)
    sampled_linspaces2 = linspaces2[:, ::fine, ...]
    sampled_sols2 = sols2[:, ::fine, ...]

    return sols, sampled_sols2
    #return sols


deltas = 1 / jnp.array([16, 32, 64, 128, 256, 512, 1024])
Ns = 1 / deltas
fineN = Ns
Mdeltas = jnp.ones((len(deltas),)) * (Ns) ** 0.
T = 10.0
Ndeltas = T / deltas

folder = "./"
solver_name = "EKF0_SSM"
problem_name = "FHN"
prefix = f"{solver_name}_{problem_name}"
for n in range(len(Ndeltas)):
    fine = int(fineN[n])
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    s1, s2 = experiment(delta, N, M, fine)
    #s1 = experiment(delta, N, M)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
# warning: no path convergence, only moment.
