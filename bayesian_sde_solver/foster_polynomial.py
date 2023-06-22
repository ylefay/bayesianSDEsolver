import math
from functools import partial

import jax
import jax.numpy as jnp


def get_approx(dim=1):
    # this gives the parabola approximation of a Brownian motion over a time interval of length dt.

    @partial(jnp.vectorize, signature="(d),()->(e),(e)")
    def parabolas(key, dt):
        eps_0, eps_1 = jax.random.normal(key, shape=(2, dim))

        eps_0 *= jnp.sqrt(dt)
        eps_1 *= jnp.sqrt(0.5 * dt)

        return eps_0, eps_1

    # @partial(jnp.vectorize, signature="(),(),(d),(d)->(d)")
    # @partial(jnp.vectorize, signature="(),(),(),()->()")
    def eval_parabola(t, dt, a, b):
        u = t / dt
        return a * u + b * math.sqrt(6) * u * (u - 1)

    return parabolas, eval_parabola
