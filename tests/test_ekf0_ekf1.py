import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf1
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.sde_solver import sde_solver


def test_gbm_ekf0():
    a = 1.
    b = 1.

    def drift(x, t):
        return a * x

    def sigma(x, t):
        return b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 50
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    def wrapped_ekf0(_key, init, vector_field, T):
        M = 200
        return ekf0(None, init=init, vector_field=vector_field, h=T / M, N=M)

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
        sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=2
    )
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=2)


def test_gbm_ekf1():
    a = 1
    b = 1

    def drift(x, t):
        return a * x

    def sigma(x, t):
        return b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    N = 50
    delta = 1 / N

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    def wrapped_ekf1(_key, init, vector_field, T):
        M = 200
        return ekf1(key=None, init=init, vector_field=vector_field, h=T / M, N=M)

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
        sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=2
    )
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=2)


def test_harmonic_oscillator_ekf0():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    gamma = 1.0
    D = 1.0
    sig = 2.0

    M = jnp.array([[0.0, 1.0], [-D, -gamma]])
    C = jnp.array([[0.0], [sig]])

    def drift(x, t):
        return jnp.dot(M, x)

    def sigma(x, t):
        return C

    x0 = jnp.ones((2,))
    N = 50
    delta = 1 / N

    def wrapped_ekf0(_key, init, vector_field, T):
        M = 200
        return ekf0(None, init=init, vector_field=vector_field, h=T / M, N=M)

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
        return sig ** 2 * jnp.array(
            [
                [1 / 3 * t ** 3, 1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma],
                [1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma, t - gamma * t ** 2 + 1 / 3 * t ** 3 * (2 * gamma ** 2 - D)],
            ]
        )

    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        theoretical_variance_up_to_order3(1),
        decimal=2,
    )


def test_harmonic_oscillator_ekf1():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    gamma = 1.0
    D = 1.0
    sig = 2.0

    M = jnp.array([[0.0, 1.0], [-D, -gamma]])
    C = jnp.array([[0.0], [sig]])

    def drift(x, t):
        return jnp.dot(M, x)

    def sigma(x, t):
        return C

    x0 = jnp.ones((2,))
    N = 50
    delta = 1 / N

    def wrapped_ekf1(_key, init, vector_field, T):
        M = 200
        return ekf1(key=None, init=init, vector_field=vector_field, h=T / M, N=M)

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

    def theoretical_mean_variance(t):
        mean = linalg.expm(M * t) @ x0

        @jax.vmap
        def integrand_var(s):
            B = linalg.expm(M * s) @ C
            return B @ B.T

        linspace_int = jnp.linspace(0, t, 1000)
        var = jnp.trapz(integrand_var(linspace_int), linspace_int, axis=0)
        return mean, var

    npt.assert_array_almost_equal(
        jnp.cov(sols[:, -1], rowvar=False),
        theoretical_mean_variance(1)[1],
        decimal=2,
    )
