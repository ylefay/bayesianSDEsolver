import jax
import jax.numpy as jnp


def solver(init, vector_field, h, N):
    """
    Euler method.
    """

    def body(x, t):
        out = x + h * vector_field(x, t)
        return out, None

    inps = jnp.linspace(0, N * h - h, N)
    y, samples = jax.lax.scan(body, init, inps)
    return y
