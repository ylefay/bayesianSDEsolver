import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0, ekf1
from bayesian_sde_solver.sde_solver import sde_solver


# tests both ekf0, ekf1 and foster polynomial
def test_gbm_ekf0():
    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 100
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    def wrapped_ekf0(_key, init, vector_field, T):
        M = 10
        return ekf0(key=_key, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped_ekf0,
        )

    linspaces, sols = wrapped_filter_parabola(keys)
    npt.assert_almost_equal(
        sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)


def test_gbm_ekf1():
    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 100
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    def wrapped_ekf1(_key, init, vector_field, T):
        M = 1
        return ekf1(key=_key, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped_ekf1,
        )

    linspaces, sols = wrapped_filter_parabola(keys)
    npt.assert_almost_equal(
        sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)


def test_harmonic_oscillator_ekf0():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    gamma = 1.0
    D = 1.0
    sig = 2.0

    drift = lambda x, t: jnp.dot(jnp.array([[0.0, 1.0], [-D, -gamma]]), x)
    sigma = lambda x, t: jnp.array([[0.0], [sig]])

    x0 = jnp.ones((2,))
    N = 100
    delta = 2 / N

    def wrapped_ekf0(_key, init, vector_field, T):
        N = 10
        return ekf0(key=None, init=init, vector_field=vector_field, h=T / N, N=N)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped_ekf0,
        )

    linspaces, sols = wrapped_filter_parabola(keys)

    def theoretical_variance_up_to_order3(k):
        t = k * delta
        return sig**2 * jnp.array(
            [
                [1 / 3 * t**3, 1 / 2 * t**2 - 1 / 2 * t**3 * gamma],
                [1 / 2 * t**2 - 1 / 2 * t**3 * gamma, t - gamma * t**2 + 1 / 3 * t**3 * (2 * gamma**2 - D)],
            ]
        )

    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        theoretical_variance_up_to_order3(1),
        decimal=2,
    )
