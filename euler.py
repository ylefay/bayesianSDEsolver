import numpy as np
import jax


def euler(init, vector_field, h, N):

    def body(x, t):
        out = x + h * vector_field(x, t)
        return out, None

    inps = np.linspace(0, N * h - h, N)
    y, _ = jax.lax.scan(body, init, inps)
    return y
