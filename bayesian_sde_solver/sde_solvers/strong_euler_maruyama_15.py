import jax
import jax.numpy as jnp
import numpy as np


def solver(key, init, drift, sigma, h, N):
    dim = sigma(init, 0.).shape[1]

    def body(x, inp):
        raise NotImplementedError
    keys = jax.random.split(key, N)
    ts = np.linspace(0, N * h - h, N)
    inps = keys, ts
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)

    return ts, samples
