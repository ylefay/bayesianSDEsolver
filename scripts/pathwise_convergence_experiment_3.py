from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise
from bayesian_sde_solver.ssm_parabola import ekf0_marginal_parabola
from bayesian_sde_solver.ssm_parabola import ssm_parabola_ode_solver

JAX_KEY = jax.random.PRNGKey(1337)

gamma = 1.0
sig = 1.0
eps = 1.0
alpha = 1.0
s = 1.0




def drift(x, t):
    def pi(x):
        return jnp.exp(-x @ x.T / 2)

    return jax.jacfwd(lambda z: jnp.log(pi(z)))(x)


def sigma(x, t):
    return jnp.array([[jnp.sqrt(2)]])


x0 = jnp.ones((1,))

x0 = jnp.ones((2,))


def drift(x, t):
    return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x


def sigma(x, t):
    return jnp.array([[0.0, 0.0], [1.0, 0.0]])

def drift(x, t):
    return jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
        [s / eps - x[0] ** 3 / eps, alpha])


def sigma(x, t):
    return jnp.array([[0.0, 0.0], [sig, 0.0]])

# stochastic pendulum
x0 = jnp.array([jnp.pi / 4, jnp.pi / 4]).reshape((2, ))


def drift(x, t):
    return jnp.array([x[1], -9.81 * jnp.sin(x[0])])


def sigma(x, t):
    return jnp.array([[0.0, 0.0], [1.0, 0.0]])
drift_s, sigma_s = to_stratonovich(drift, sigma)

init = x0


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s)", excluded=(1, 2,))
@partial(jax.jit, static_argnums=(1, 2,))
def experiment(delta, N, M):
    keys = jax.random.split(JAX_KEY, 1_000_000)

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
    incs = coeffs[0]
    shape_incs = incs.shape
    incs *= jnp.sqrt(1 / delta)

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        return euler_maruyama_pathwise(inc, init=x0, drift=drift, sigma=sigma, h=delta, N=1 * shape_incs[1])

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    return sols, sols2



Ns = jnp.array([4, 8, 16, 32, 64, 128, 256])
deltas = jnp.array([0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])
Mdeltas = jnp.ones((len(deltas),)) * Ns ** 0
Ndeltas = Ns

folder = "./"
solver_name = "ekf0_ssm_"
problem_name = "pendulum"
prefix = solver_name + "_" + problem_name
for n in range(len(Ndeltas)):
    fine = 1  # since we sample the parabola, we cannot go with finer solutions
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    s1, s2 = experiment(delta, N, M)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
