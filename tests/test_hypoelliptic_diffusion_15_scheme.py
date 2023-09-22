import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import hypoelliptic_diffusion_15_scheme
from bayesian_sde_solver.utils.ivp import harmonic_oscillator_square, square_matrix_fhn, synaptic_conductance


def test_harmonic_oscillator():
    x0, drift, sigma, _, _, local_theoretical_variance = harmonic_oscillator_square()

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N
        )

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    N = 100
    h = 2 / N

    linspaces, sols = wrapped_hypoelliptic_15(keys)
    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        local_theoretical_variance(h),
        decimal=2,
    )


def test_fitzhugh_nagumo():
    x0, _drift, _sigma, local_theoretical_mean, local_theoretical_variance = square_matrix_fhn()
    drift = lambda x: _drift(x, 0.0)
    sigma = lambda x: _sigma(x, 0.0)

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N
        )

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    N = 10000
    h = 10 / N

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        local_theoretical_variance(h),
        decimal=2,
    )
    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), local_theoretical_mean(h), decimal=2
    )


def test_synaptic_conductance():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)
    N = 1000
    h = 1 / N
    x0, drift, sigma, theoretical_mean_up_to_order_2, theoretical_variance_up_to_order_3_first_coordinate = synaptic_conductance()

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return hypoelliptic_diffusion_15_scheme(
            key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N
        )

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), theoretical_mean_up_to_order_2(h), decimal=2
    )
    npt.assert_almost_equal(
        sols[:, 4, 0].var(),
        theoretical_variance_up_to_order_3_first_coordinate(4 * h),
        decimal=5,
    )
