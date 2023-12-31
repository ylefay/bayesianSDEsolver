from functools import partial

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf1, ekf0, ekf0_2, ekf1_2
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.utils.ivp import harmonic_oscillator

SOLVERS = [ekf0, ekf1, ekf1_2, ekf0_2, partial(ekf0, sqrt=True),
           partial(ekf1, sqrt=True), partial(ekf0_2, sqrt=True),
           partial(ekf1_2, sqrt=True)]

N = 1000
M = 5
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
    if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, x0, P0)

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

    linspaces, sols, *_ = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
        sols = sols[0]
    npt.assert_allclose(sols[:, -1].mean(axis=0), x0 * jnp.exp(mu * delta * N), rtol=10e-02)
    npt.assert_allclose(sols[:, -1].std(axis=0),
                        (0.5 * (jnp.exp(sig * N * delta * 2) - 1)) ** 0.5, rtol=10e-02)


@pytest.mark.parametrize("solver", SOLVERS)
def test_harmonic_oscillator(solver):
    x0, drift, sigma, theoretical_mean, theoretical_variance = harmonic_oscillator()
    init = x0
    if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
        P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
        init = (x0, x0, P0)

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

    linspaces, sols, *_ = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
        sols = sols[0]

    m, cov = theoretical_mean(N * delta), theoretical_variance(N * delta)
    npt.assert_allclose(
        sols[:, -1].mean(axis=0),
        m,
        rtol=10e-02
    )
    if sigma(x0, 0.0)[0, 1] != 0:
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
        if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
            P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
            init = (x0, x0, P0)
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

        linspaces, sols, *_ = wrapped_filter_parabola(keys)
        if solver in [ekf0_2, ekf1_2] or isinstance(solver, partial) and solver.func in [ekf0_2, ekf1_2]:
            sols = sols[0]
        return sols[:, -1].mean(axis=0), sols[:, -1].std(axis=0)

    res = [test(solver) for solver in SOLVERS]
    npt.assert_almost_equal(jnp.sum(jnp.abs(jnp.array([res[i][0] - res[0][0] for i in range(len(SOLVERS))])), axis=0),
                            jnp.zeros((1,)), decimal=3)
    npt.assert_almost_equal(jnp.sum(jnp.abs(jnp.array([res[i][1] - res[0][1] for i in range(len(SOLVERS))])), axis=0),
                            jnp.zeros((1,)), decimal=3)
