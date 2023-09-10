from functools import partial

import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx as _get_approx
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.sde_solver import sde_solver
from bayesian_sde_solver.ssm_parabola import ekf0_marginal_parabola, ekf1_marginal_parabola
from bayesian_sde_solver.ssm_parabola import ssm_parabola_ode_solver

JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)

ekf0 = partial(ekf0, sqrt=True)


def test_linear_sde_unidimensional():
    delta = 0.001
    N = int(1 / delta)

    def drift(x, t):
        return x

    def sigma(x, t):
        return jnp.array([[1.0]])

    x0 = jnp.ones((1,))

    def solver(key, init, drift, diffusion, T):
        return ekf0_marginal_parabola(key, init, delta=T, drift=drift, diffusion=diffusion, h=T, N=1, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, delta=delta, N=N,
                                       solver=solver)
    _, sols = wrapped_filter_parabola(keys)

    # EKF solution
    def wrapped2(_key, init, vector_field, T):
        return ekf0(None, init=init, vector_field=vector_field, h=T / 1, N=1)

    npt.assert_allclose(sols[:, -1].mean(axis=0), x0 * jnp.exp(1.0 * delta * N), rtol=10e-02)
    npt.assert_allclose(sols[:, -1].std(axis=0),
                        (0.5 * (jnp.exp(1.0 * N * delta * 2) - 1)) ** 0.5, rtol=10e-02)

    @jax.vmap
    def wrapped_filter_parabola2(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=_get_approx,
            delta=delta,
            N=N,
            ode_int=wrapped2,
        )

    _, sols2, *_ = wrapped_filter_parabola2(keys)
    npt.assert_allclose(sols2[:, -1].mean(axis=0), sols[:, -1].mean(axis=0), rtol=10e-02)
    npt.assert_allclose(jnp.cov(sols2[:, -1], rowvar=False), jnp.cov(sols[:, -1], rowvar=False), rtol=10e-02)


def test_multidimensional_sde():
    delta = 0.01
    N = int(1 / delta)
    get_approx = partial(_get_approx, dim=2)

    def drift(x, t):
        return x

    def sigma(x, t):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    x0 = jnp.ones((2,))

    def solver(key, init, drift, diffusion, T):
        return ekf0_marginal_parabola(key, init, delta=T, drift=drift, diffusion=diffusion, h=T, N=1, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, delta=delta, N=N,
                                       solver=solver)

    linspace1, sols = wrapped_filter_parabola(keys)

    npt.assert_allclose(sols[:, -1].mean(axis=0), x0 * jnp.exp(N * delta), rtol=10e-02)
    npt.assert_allclose(jnp.cov(sols[:, -1], rowvar=False),
                        jnp.array([[x0[0] ** 2 * (0.5 * (jnp.exp(1.0 * N * delta * 2) - 1)), 0],
                                   [0, x0[1] ** 2 * (0.5 * (jnp.exp(1.0 * N * delta * 2) - 1))]])

                        , atol=10e-01)


def test_ibm():
    delta = 0.01
    N = int(1 / delta)
    get_approx = partial(_get_approx, dim=2)

    def drift(x, t):
        return jnp.array([[0., 1.0], [0., 0.]]) @ x

    def sigma(x, t):
        return jnp.array([[0., 0.], [1.0, 0.]])

    x0 = jnp.ones((2,))

    def solver(key, init, drift, diffusion, T):
        return ekf0_marginal_parabola(key, init, delta=T, drift=drift, diffusion=diffusion, h=T, N=1, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, delta=delta, N=N,
                                       solver=solver)

    linspace1, sols = wrapped_filter_parabola(keys)
    T = N * delta
    npt.assert_allclose(sols[:, -1].mean(axis=0),
                        jnp.array([x0[0] + x0[1] * T, x0[1]])
                        , rtol=10e-02)
    npt.assert_allclose(jnp.cov(sols[:, -1], rowvar=False),
                        jnp.array([[T ** 3 / 3, T ** 2 / 2],
                                   [T ** 2 / 2, T]])

                        , rtol=10e-02)
