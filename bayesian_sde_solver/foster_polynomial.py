from functools import partial

import jax
import jax.numpy as jnp


def get_approx(dim=1):
    """
    This gives a sampled parabola approximation over a time interval of length dt.
    """

    @partial(jnp.vectorize, signature="(d),()->(e),(e)")
    def parabolas(key, dt):
        eps_0, eps_1 = jax.random.normal(key, shape=(2, dim))

        eps_0 *= jnp.sqrt(dt)
        eps_1 *= jnp.sqrt(0.5 * dt)

        return eps_0, eps_1

    @partial(jnp.vectorize, signature="(),(),(),()->()")
    def eval_parabola(t, dt, a, b):
        u = t / dt
        return a * u + b * jnp.sqrt(6) * u * (
                    1 - u)  # Instead of * (u - 1) as given in Foster's thesis. However, it is a matter of convention.

    return parabolas, eval_parabola


def get_approx_and_brownian(dim=1, N=1000):
    """
    This gives the parabola approximation of a fine sampled Brownian motion, as well as the corresponding Brownian motion.
    """

    @partial(jnp.vectorize, signature="(d),()->(e),(e),(f,s)")
    def parabolas(key, dt):
        incs = jax.random.normal(key, shape=(N, dim))
        bm = jnp.cumsum(incs, axis=0)
        bm *= jnp.sqrt(dt / N)

        _is = jnp.arange(1, N + 1)

        eps_0 = bm.at[-1].get()

        @jax.vmap
        def integrand(i):
            return (bm.at[i].get()) * jnp.sqrt(6)

        eps_1 = jnp.trapz(integrand(_is), dx=1 / N, axis=0) - eps_0 * 0.5 * jnp.sqrt(6)

        return eps_0, eps_1, incs

    def eval_parabola(t, dt, a, b, _):
        _, _eval_parabola = get_approx(dim=dim)
        return _eval_parabola(t, dt, a, b)

    return parabolas, eval_parabola


def get_approx_fine(dim=1, N=100):
    """
    This gives the parabola approximation constructed using fine parabolas.
    This method is used by Foster to compute pathwise errors.
    """
    _parabolas, _eval_parabola = get_approx(dim=dim)

    @partial(jnp.vectorize, signature="(d),()->(e),(e),(f,s),(f,s)")
    def parabolas(key, dt):
        fine_dt = dt / N
        keys = jax.random.split(key, N)
        fine_eps_0s, fine_eps_1s = _parabolas(keys, fine_dt)
        fine_eps_1s *= 1 / jnp.sqrt(6)

        # See https://github.com/james-m-foster/igbm-simulation/blob/master/igbm.cpp, l209-230

        def update_eps(carry, inps):
            eps_0, eps_1 = carry
            fine_eps_0, fine_eps_1 = inps
            eps_1 += fine_dt * (eps_0 + 0.5 * fine_eps_0 + fine_eps_1)
            eps_0 += fine_eps_0
            return (eps_0, eps_1), None

        res = jax.lax.scan(update_eps, (jnp.zeros((dim,)), jnp.zeros((dim,))), (fine_eps_0s, fine_eps_1s))
        eps_0, eps_1 = res[0]
        eps_1 = eps_1 * 1 / dt - eps_0 * 0.5
        eps_1 *= jnp.sqrt(6)
        return eps_0, eps_1, fine_eps_0s, fine_eps_1s

    def eval_parabola(t, dt, a, b, _, __):
        return _eval_parabola(t, dt, a, b)

    return parabolas, eval_parabola
