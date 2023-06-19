from functools import partial

import jax.numpy as jnp
import jax.random
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx


def test_moments():
    # this tests that the parabola approximation has mean 0 and variance h at time h.
    N = 1_000
    h = 1 / N

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, 100)

    get_coeffs, eval_fn = parabola_approx()

    @jax.vmap
    def get_increments(key):
        coeffs = get_coeffs(key, h, 1)
        return eval_fn(h, h, *coeffs)

    increments = get_increments(keys)
    npt.assert_almost_equal(increments.mean(), 0, decimal=2)
    npt.assert_almost_equal(increments.std(), h ** 0.5, decimal=2)


def test_path_integral():
    # this tests that the integral of u * dWu has mean 0 and variance h ** 3 / 3 identity,
    # where dWu is the parabola approximation.
    N, M = 500, 10_000
    h = 0.5

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, M)

    get_coeffs, eval_fn = parabola_approx()
    coeffs = jax.vmap(get_coeffs, in_axes=(0, None, None))(keys, h, 2)
    linspace = jnp.linspace(0, h, N + 1)

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def integrand_mean(t, a, b):
        func = lambda z: eval_fn(z, h, a, b)
        return t * jax.jacfwd(func)(t)

    ys = integrand_mean(linspace, *coeffs)
    # trapz = jax.vmap(jnp.trapz, in_axes=[[1, 2], None])(ys, linspace) #uni
    trapz = jax.vmap(jax.vmap(jnp.trapz, in_axes=[1, None]), in_axes=[2, None])(ys, linspace)
    # trapz = jnp.trapz(ys, linspace, axis=0) #works

    npt.assert_array_almost_equal(trapz.mean(axis=1), jnp.array([0.0, 0.0]), decimal=2)
    npt.assert_array_almost_equal(jnp.cov(trapz), h ** 3 / 3 * jnp.identity(2), decimal=2)
