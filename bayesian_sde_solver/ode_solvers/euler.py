import jax
import numpy as np


def solver(init, vector_field, h, N):
    def body(x, t):
        out = x + h * vector_field(x, t)
        return out, None

    inps = np.linspace(0, N * h - h, N)
    y, samples = jax.lax.scan(body, init, inps)
    return y
