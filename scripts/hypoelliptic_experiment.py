from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0, ekf1, ieks, euler
from bayesian_sde_solver.sde_solver import sde_solver

JAX_KEY = jax.random.PRNGKey(1337)


@partial(jnp.vectorize, signature='()->(s,s)', excluded=(1, ))
@partial(jax.jit, static_argnums=(1, ))
def ibm(delta, N=20):
    keys = jax.random.split(JAX_KEY, 1_000_00)

    drift = lambda x, t: jnp.dot(jnp.array([[0.0, 1.0], [0.0, 0.0]]), x)
    sigma = lambda x, t: jnp.array([[0.0], [1.0]])

    x0 = jnp.ones((2,))

    def wrapped(_key, init, vector_field, T):
        return ekf0(key=None, init=init, vector_field=vector_field, h=T / N, N=N)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                          ode_int=wrapped)

    _, sols = wrapped_filter_parabola(keys)
    return jnp.cov(sols[:, 1], rowvar=False)


from bayesian_sde_solver.sde_solvers import hypoelliptic_diffusion_15_scheme


@partial(jnp.vectorize, signature='()->(d,d)')
def ibm_15(delta):
    keys = jax.random.split(JAX_KEY, 1_000_00)

    drift = lambda x: jnp.dot(jnp.array([[0.0, 1.0], [0.0, 0.0]]), x)
    sigma = lambda x: jnp.array([[0.0], [1.0]])

    x0 = jnp.ones((2,))
    N = 1

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=delta, N=N)

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    return jnp.cov(sols[:, 1], rowvar=False)


deltas = jnp.logspace(-3, 0, 20)
import numpy as np
Ndeltas = np.ceil(1 / deltas)
ibms = jnp.empty(shape=(2, 2))
ibms = jnp.stack([ibm(delta, int(N))[0, 0]/delta**3 for delta, N in zip(deltas, Ndeltas)])

#ibms = ibm(deltas)
ibms_15 = ibm_15(deltas)
print(ibms)
plt.plot(1 / deltas, ibms)
plt.plot(1 / deltas, ibms_15[::-1, 0, 0] / deltas[::-1] ** 3, label="Ditlevsen & Samson")
plt.legend()
plt.show()
