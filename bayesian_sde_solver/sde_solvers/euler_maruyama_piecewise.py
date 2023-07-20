import jax
import jax.numpy as jnp
import numpy as np


def solver(incs, init, drift, sigma, h, N):
    def body(x, inp):
        inc, t = inp
        dW = h ** 0.5 * inc
        out = x + h * drift(x, t) + sigma(x, t) @ dW
        return out, out

    ts = np.linspace(0, N * h, N + 1)
    inps = incs, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples
