from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ode_solvers import ekf0, ekf1, ieks, euler
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.ito_stratonovich import to_stratonovich

JAX_KEY = jax.random.PRNGKey(1337)


@partial(jnp.vectorize, signature='()->(d)', excluded=(1, 2, ))
@partial(jax.jit, static_argnums=(1, 2))
def gbm(delta, N=20, M=20):
    keys = jax.random.split(JAX_KEY, 1_000_00)

    drift = lambda x, t: x
    sigma = lambda x, t: jnp.eye(1).reshape((1, 1))

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))

    def wrapped(_key, init, vector_field, T):
        return ekf0(key=None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N,
                          ode_int=wrapped)

    _, sols = wrapped_filter_parabola(keys)
    return jnp.array([jnp.cov(sols[:, -1], rowvar=False), jnp.mean(sols[:, -1])])


deltas = jnp.logspace(-4, -4, 1)
Ndeltas = np.ceil(1/deltas)
Mdeltas = np.ceil(1/deltas**0.0)
ibms = jnp.empty(shape=(1, 1))
for delta, N, M in zip(deltas, Ndeltas, Mdeltas):
    print(f"{delta}, {int(N)}, {int(M)}")
ibms = jnp.stack([gbm(delta, int(N), int(M)) for delta, N, M in zip(deltas, Ndeltas, Mdeltas)])
ts = jnp.stack([delta * int(N) for delta, N in zip(deltas, Ndeltas)])
plt.semilogy(1/deltas, jnp.exp(2*ts)*(jnp.exp(ts)-1))
#ibms = ibm(deltas)
print("deltas")
print(deltas)
print("ts")
print(ts)
#print(jnp.exp(2*ts)*(jnp.exp(ts)-1)) #var gbm
#print(jnp.exp(ts)) #mean gbm
print("pred")
print(0.5*(jnp.exp(2*ts)-1)) #var Xdt + dWt = dXt, Xt = e^(t)(X0 + int e^(-s)dWs),
print(jnp.exp(ts)) #mean Xt
print("res")
print(ibms)
#plt.semilogy(1 / deltas, ibms)
#plt.legend()
#plt.show()
