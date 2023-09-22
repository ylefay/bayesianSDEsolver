import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import euler
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.utils.ivp import gbm, harmonic_oscillator

JAX_KEY = jax.random.PRNGKey(1337)
M = 100  # euler pts
N = 100
keys = jax.random.split(JAX_KEY, 1_000)


def test_gbm_euler():
    x0, drift, sigma, mean, var = gbm()

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

    linspaces, sols, *_ = wrapped_parabola(keys)
    npt.assert_almost_equal(
        sols[:, -1].std(axis=0), var(1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(axis=0), mean(1), decimal=1)


def test_harmonic_oscillator_euler():
    x0, drift, sigma, _, theoretical_variance = harmonic_oscillator()
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

    linspaces, sols, *_ = wrapped_parabola(keys)

    npt.assert_allclose(
        jnp.cov(sols[:, 1], rowvar=False),
        theoretical_variance(delta),
        rtol=10e-02
    )
