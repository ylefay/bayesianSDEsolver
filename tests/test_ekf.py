import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import numpy.testing as npt
import pytest

from functools import partial

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf1, ekf0, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver

SOLVERS = [ekf0, ekf1, ekf1_2]
SOLVERS = [ekf0]
_len = len(SOLVERS)
for solver in SOLVERS[:_len]:
    if solver not in [ekf1_2]:
        SOLVERS.append(partial(solver, sqrt=True))
N = 1000
M = 100
JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)


@pytest.mark.parametrize("solver", SOLVERS)
def test(solver):
    mu = 1.0
    sig = 1.0

    def drift(x, t):
        return mu * x

    def sigma(x, t):
        return jnp.array([[sig]])  # jnp.diag(x) does not work, multiplicative noise?

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
    npt.assert_allclose(sols[:, -1].std(axis=0),
                        (0.5 * (jnp.exp(sig * N * delta * 2) - 1))**0.5, rtol=10e-02)


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

    m, cov = theoretical_mean_variance(N * delta)
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


def test_all_agree():
    def drift(x, t):
        return (t + 1) * x

    def sigma(x, t):
        return jnp.array([[1.0]]) * (t + 1)

    def test(solver):
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
        return sols[:, -1].mean(axis=0), sols[:, -1].std(axis=0)

    res = [test(solver) for solver in SOLVERS]
    npt.assert_almost_equal(jnp.sum(jnp.abs(jnp.array([res[i][0] - res[0][0] for i in range(len(SOLVERS))])), axis=0),
                            jnp.zeros((1,)), decimal=5)
    npt.assert_almost_equal(jnp.sum(jnp.abs(jnp.array([res[i][1] - res[0][1] for i in range(len(SOLVERS))])), axis=0),
                            jnp.zeros((1,)), decimal=5)
