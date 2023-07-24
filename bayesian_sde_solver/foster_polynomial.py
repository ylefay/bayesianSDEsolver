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

    @partial(jnp.vectorize, signature="(),(),(),()->()")
    def eval_parabola(t, dt, a, b):
        u = t / dt
        return a * u + b * jnp.sqrt(6) * u * (u - 1)

    return parabolas, eval_parabola


def get_approx_and_brownian(dim=1, N=15000):
    # this gives the parabola approximation of a Brownian motion, as well as the corresponding Brownian motion.

    def parabolas(key, dt):
        incs = jax.random.normal(key, shape=(N, dim))
        bm = jnp.cumsum(incs, axis=0)
        bm *= jnp.sqrt(dt / N)

        _is = jnp.arange(1, N + 1)

        eps_0 = bm.at[-1].get()

        @jax.vmap
        def integrand(i):
            u = (i + 1) / N
            return - (bm.at[i].get() - eps_0 * u) * jnp.sqrt(6)

        eps_1 = jnp.trapz(integrand(_is), dx=1 / N, axis=0)

        return eps_0, eps_1, incs

    def eval_parabola(t, dt, a, b, _):
        _, _eval_parabola = get_approx(dim=dim)
        return _eval_parabola(t, dt, a, b)

    return parabolas, eval_parabola
