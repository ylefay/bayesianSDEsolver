import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import euler
from bayesian_sde_solver.sde_solver import sde_solver

JAX_KEY = jax.random.PRNGKey(1337)
M = 100  # euler pts
N = 100
keys = jax.random.split(JAX_KEY, 1_000)
def test_gbm_euler():

    a = 1
    b = 1

    def drift(x, t):
        return a * x

    def sigma(x, t):
        return b * jnp.diag(x)

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    delta = 1 / N


    def wrapped_euler(_key, init, vector_field, T):
        return euler(init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped_euler,
        )

    linspaces, sols = wrapped_parabola(keys)
    npt.assert_almost_equal(
        sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(axis=0), x0 * jnp.exp(a), decimal=1)


def test_harmonic_oscillator_euler():
    gamma = 1.0
    D = 1.0
    sig = 2.0

    drift = lambda x, t: jnp.dot(jnp.array([[0.0, 1.0], [-D, -gamma]]), x)
    sigma = lambda x, t: jnp.array([[0.0], [sig]])

    x0 = jnp.ones((2,))
    N = 1000
    delta = 1 / N

    M = 100

    def wrapped_euler(_key, init, vector_field, T):
        return euler(init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=lambda: parabola_approx(1),
            delta=delta,
            N=N,
            ode_int=wrapped_euler,
        )

    linspaces, sols = wrapped_parabola(keys)

    def theoretical_variance_up_to_order3(t):
        return sig ** 2 * jnp.array(
            [
                [1 / 3 * t ** 3, 1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma],
                [1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma, t - gamma * t ** 2 + 1 / 3 * t ** 3 * (2 * gamma ** 2 - D)]
            ]
        )

    npt.assert_allclose(
        jnp.cov(sols[:, 1], rowvar=False),
        theoretical_variance_up_to_order3(delta),
        rtol=10e-02
    )
