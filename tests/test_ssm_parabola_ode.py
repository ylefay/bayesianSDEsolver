import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.foster_polynomial import get_approx
from bayesian_sde_solver.ssm_parabola import ekf0_marginal_parabola
from bayesian_sde_solver.ssm_parabola import ssm_parabola_ode_solver
from bayesian_sde_solver.sde_solver import sde_solver
from functools import partial

from bayesian_sde_solver.ode_solvers import ekf0

JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)

ekf0 = partial(ekf0, sqrt=True)


def test():
    # SSM parabola ode test
    # For linear SDEs, should exactly coincide with simple ekf methods
    delta = 1.0
    N = int(1 / delta)

    def drift(x, t):
        return x

    def sigma(x, t):
        return jnp.array([[1.0]])

    x0 = jnp.ones((1,))

    def solver(key, init, delta, drift, diffusion, T):
        return ekf0_marginal_parabola(None, init, delta, drift, diffusion, h=T, N=1, sqrt=True)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return ssm_parabola_ode_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, bm=get_approx, delta=delta, N=N,
                                       solver=solver)

    with jax.disable_jit():
        linspace1, sols, *_ = wrapped_filter_parabola(keys)

    # EKF solution
    def wrapped2(_key, init, vector_field, T):
        return ekf0(None, init=init, vector_field=vector_field, h=T / 1, N=1)

    @jax.vmap
    def wrapped_filter_parabola2(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=x0,
            bm=get_approx,
            delta=delta,
            N=N,
            ode_int=wrapped2,
        )

    with jax.disable_jit():
        linspace2, sols2, *_ = wrapped_filter_parabola2(keys)

    npt.assert_allclose(sols[:, -1].mean(axis=0), x0 * jnp.exp(1.0 * delta * N), rtol=10e-02)
    npt.assert_allclose(sols[:, -1].std(axis=0),
                        (0.5 * (jnp.exp(1.0 * N * delta * 2) - 1)) ** 0.5, rtol=10e-02)
