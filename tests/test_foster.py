from functools import partial

import jax.numpy as jnp
import jax.random
import numpy.testing as npt
import pytest

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.foster_polynomial import get_approx_and_brownian as parabola_approx_and_brownian
from bayesian_sde_solver.foster_polynomial import get_approx_fine

PARABOLA_APPROXIMATIONS = [parabola_approx, parabola_approx_and_brownian, get_approx_fine]


@pytest.mark.parametrize("approximation", PARABOLA_APPROXIMATIONS)
def test_moments(approximation):
    # this tests that the parabola approximation has mean 0 and variance h at time h.
    N = 1_000
    h = 1 / N

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, 100)

    get_coeffs, eval_fn = approximation()

    @jax.vmap
    def get_increments(key):
        coeffs = get_coeffs(key, h)
        return eval_fn(h, h, *coeffs)

    increments = get_increments(keys)
    npt.assert_almost_equal(increments.mean(axis=0), 0, decimal=2)
    npt.assert_almost_equal(increments.std(axis=0), h ** 0.5, decimal=2)


@pytest.mark.parametrize("approximation", PARABOLA_APPROXIMATIONS)
def test_path_integral(approximation):
    # this tests that the integral of u * dWu has mean 0 and variance h ** 3 / 3 identity,
    # where dWu is the parabola approximation.
    dim = 2
    N, M = 500, 10_000
    h = 0.5

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, M)

    get_coeffs, eval_fn = approximation(dim)
    coeffs = jax.vmap(get_coeffs, in_axes=(0, None))(keys, h)

    linspace = jnp.linspace(0, h, N + 1)

    @partial(jax.vmap, in_axes=(0, None))
    def integrand_mean(t, coeffs):
        func = lambda z: eval_fn(z, h, *coeffs)
        return t * jax.jacfwd(func)(t)

    ys = integrand_mean(linspace, coeffs)
    trapz = jnp.trapz(ys, linspace, axis=0)

    npt.assert_array_almost_equal(trapz.mean(axis=0), jnp.zeros((dim,)), decimal=2)
    npt.assert_array_almost_equal(
        jnp.cov(trapz, rowvar=False), h ** 3 / 3 * jnp.identity(dim), decimal=3
    )
