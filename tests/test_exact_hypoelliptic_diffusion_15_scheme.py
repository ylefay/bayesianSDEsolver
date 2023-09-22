import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import diag_15_scheme
from bayesian_sde_solver.utils.ivp import synaptic_conductance, harmonic_oscillator_square


def test_synaptic_conductance():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)
    N = 100
    h = 1 / N
    x0, drift, sigma, theoretical_mean, theoretical_variance = synaptic_conductance()

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), theoretical_mean(h), decimal=2
    )
    npt.assert_almost_equal(
        sols[:, 10, 0].var(),
        theoretical_variance(10 * h),
        decimal=4,
    )


def test_harmonic_oscillator():
    x0, drift, sigma, theoretical_mean, theoretical_variance, local_theoretical_variance = harmonic_oscillator_square()

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    N = 100
    h = 1 / N

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    npt.assert_allclose(
        sols[:, -1].mean(axis=0),
        theoretical_mean(N * h),
        rtol=10e-02
    )
    if sigma(x0)[1, 0] != 0:
        npt.assert_allclose(
            jnp.cov(sols[:, -1], rowvar=False),
            theoretical_variance(N * h),
            rtol=10e-02
        )
    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        local_theoretical_variance(h),
        decimal=2,
    )
