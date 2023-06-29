import jax.numpy as jnp
from math import factorial
def transition_function(theta, sigma, q, dt):
    F = jnp.block([[jnp.zeros(q).reshape((q, 1)), jnp.diag(jnp.ones(q))],
                   [jnp.append(jnp.zeros(q), jnp.array([-theta])).reshape((1, q+1))]])
    A = jnp.exp(F * dt)
    Q = sigma**2*jnp.array([[dt**(2*q+1-i-j)/((2*q+1-i-j)*factorial((q-i))*factorial((q-j))) for j in range(q+1)] for i in range(q+1)])
    m = jnp.zeros(q+1)
    def transition(x):
        return jnp.dot(A, x)

    return transition, m, Q