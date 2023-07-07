import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import numpy.testing as npt
import pytest

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf1, ekf0, ekf1_2, euler
from bayesian_sde_solver.sde_solver import sde_solver

SOLVERS = [ekf0, ekf1, ekf1_2]
N = 1000
M = 1
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)


@pytest.mark.parametrize("solver", SOLVERS)
def test(solver):
    mu = 1.0
    sig = 1.0

    def drift(x, t):
        return mu * x

    def sigma(x, t):
        return jnp.array([[sig]]) #jnp.diag(x) does not work, multiplicative noise?

    drift, sigma = to_stratonovich(drift, sigma)

    x0 = jnp.ones((1,))
    init = x0
    if solver in [ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, P0)

    delta = 1 / N


    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )
    linspaces, sols = wrapped_filter_parabola(keys)
    if solver in [ekf1_2]:
        sols = sols[0]
    npt.assert_allclose(sols[:, -1].mean(axis=0), x0 * jnp.exp(mu * delta * N), rtol=10e-02)
    if sig != 0:
        npt.assert_allclose(sols[:, -1].std(), x0 * jnp.exp(mu * delta * N) * (jnp.exp(sig ** 2 * delta * N) - 1) ** 0.5, rtol=10e-02)


@pytest.mark.parametrize("solver", SOLVERS)
def test_harmonic_oscillator(solver):
    gamma = 1.0
    D = 1.0
    sig = 2.0

    Mm = jnp.array([[0.0, 1.0], [-D, -gamma]])
    C = jnp.array([[0.0], [sig]])

    def drift(x, t):
        return jnp.dot(Mm, x)

    def sigma(x, t):
        return C

    x0 = jnp.ones((2,))
    init = x0
    if solver in [ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, P0)

    delta = 1 / N

    def wrapped(_key, init, vector_field, T):
        return solver(None, init=init, vector_field=vector_field, h=T / M, N=M)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=parabola_approx,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols = wrapped_filter_parabola(keys)
    if solver in [ekf1_2]:
        sols = sols[0]

    def theoretical_mean_variance(t):
        mean = linalg.expm(Mm * t) @ x0

        @jax.vmap
        def integrand_var(s):
            B = linalg.expm(Mm * s) @ C
            return B @ B.T

        linspace_int = jnp.linspace(0, t, 1000)
        var = jnp.trapz(integrand_var(linspace_int), linspace_int, axis=0)
        return mean, var

    m, cov = theoretical_mean_variance(delta*N)
    npt.assert_allclose(
        sols[:, -1].mean(axis=0),
        m,
        rtol=10e-02
    )
    if sig != 0:
        npt.assert_allclose(
            jnp.cov(sols[:, -1], rowvar=False),
            cov,
            rtol=10e-02
        )
