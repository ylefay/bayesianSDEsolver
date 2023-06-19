import jax
import jax.numpy as jnp
import numpy as np


def solver(key, init, drift, sigma, h, N):
    dim = sigma(init, 0.).shape[1]

    def body(x, inp):
        key_k, t = inp
        bm_key = jax.random.split(key_k, 1)
        dW = h ** 0.5 * jax.random.normal(bm_key, shape=(dim, ))
        out = x + h * drift(x, t) + sigma(x, t) @ dW
        return out, out

    keys = jax.random.split(key, N)
    ts = np.linspace(0, N * h - h, N)
    inps = keys, ts
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples
