from functools import partial

import jax.numpy as jnp
import jax.random
import numpy as np
import numpy.testing as npt

from bayesian_ode_solver.foster_polynomial import get_approx as parabola_approx


def test_moments():
    N = 1_000
    h = 1 / N

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, 100)

    get_coeffs, eval_fn = parabola_approx()

    @jax.vmap
    def get_increments(key):
        coeffs = get_coeffs(key, h)
        return eval_fn(h, h, *coeffs)

    increments = get_increments(keys)
    npt.assert_almost_equal(increments.mean(), 0, decimal=2)
    npt.assert_almost_equal(increments.std(), h ** 0.5, decimal=2)


def test_path_integral():
    # this tests that the integral of u * dWu has mean 0 and variance 1/3,
    # where dWu is the parabola approximation.
    N, M = 500, 10_000
    h = 0.5

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, M)

    get_coeffs, eval_fn = parabola_approx()
    coeffs = jax.vmap(get_coeffs, in_axes=(0, None))(keys, h)
    linspace = jnp.linspace(0, h, N)

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def integrand_mean(t, a, b):
        func = lambda z: eval_fn(z, h, a, b)
        return t * jax.grad(func)(t)

    ys = integrand_mean(linspace, *coeffs)
    trapz = jax.vmap(jnp.trapz, in_axes=[1, None])(ys, linspace)

    npt.assert_almost_equal(trapz.mean(), 0, decimal=2)
    npt.assert_almost_equal(trapz.var(), h / 3, decimal=2)
