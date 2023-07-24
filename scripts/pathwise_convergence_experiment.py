import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_and_brownian as parabola_approx_and_brownian
from bayesian_sde_solver.ode_solvers import ekf0 as solver
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.sde_solvers import euler_maruyama_piecewise
import matplotlib.pyplot as plt

Ns = [1000]
M = 1
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 25)

gamma = 1.0
D = 1.0
sig = 2.0

Mm = jnp.array([[0.0, 1.0], [-D, -gamma]])
C = jnp.array([[0.0], [sig]])


def drift(x, t):
    return jnp.dot(Mm, x)


def sigma(x, t):
    return C


T = 1
x0 = jnp.ones((2,))
init = x0


def experiment(N):
    delta = T / N

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=parabola_approx_and_brownian,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols, *coeffs, incs = wrapped_filter_parabola(keys)
    shape_incs = incs.shape
    incs = incs.reshape((shape_incs[0], shape_incs[1] * shape_incs[2], shape_incs[3]))

    @jax.vmap
    def wrapped_euler_maruyama_piecewise(inc):
        P = shape_incs[1] * shape_incs[2]
        h = T / P
        return euler_maruyama_piecewise(inc, init=x0, drift=drift, sigma=sigma, h=h, N=P)

    linspaces2, sols2 = wrapped_euler_maruyama_piecewise(incs)
    sampled_linspaces2 = linspaces2[:, ::shape_incs[2], ...]
    sampled_sols2 = sols2[:, ::shape_incs[2], ...]
    return linspaces, sols, sampled_linspaces2, sampled_sols2

eps = jnp.zeros((len(Ns),))
for i, N in enumerate(Ns):
    _, s1, _, s2 = experiment(N)
    _eps = jnp.mean(jnp.max((jnp.sum((jnp.abs(s1-s2))[...,:]**2, axis=-1))**0.5, axis=-1), axis=0)
    eps = eps.at[i].set(_eps*N)
#plt.plot(Ns, eps)
#plt.show()
print(eps)
