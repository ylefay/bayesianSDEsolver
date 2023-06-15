import jax.random
import numpy as np

def parabolas(dt, N, key):
    key = jax.random.PRNGKey(key)
    key, *subkeys = jax.random.split(key, 3)
    coeffs = np.sqrt(dt) * np.array([jax.random.normal(key=subkeys[0], shape=(N+1, )), 1/np.sqrt(2)*jax.random.normal(key=subkeys[1], shape=(N+1, ))]) #increments and H_{tk,tk+1}
    e0 = lambda t: t
    e1 = lambda t: np.sqrt(6)*t*(t-1)
    return [lambda t: e0(t/dt)*coeffs[0][i] + e1(t/dt)*coeffs[1][i] for i in range(N)]

