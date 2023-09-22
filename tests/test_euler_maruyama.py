import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import euler_maruyama, euler_maruyama_pathwise
from bayesian_sde_solver.utils.ivp import ibm, gbm


def test_gbm_moment():
    # this tests the moments of the EM method for the geometric brownian motion sde.

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)
    x0, drift, sigma, theoretical_mean, theoretical_variance = gbm()

    N = 100
    h = 1.0 / N

    @jax.vmap
    def wrapped_euler_maruyama(key_op):
        return euler_maruyama(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    linspaces, sols = wrapped_euler_maruyama(keys)

    npt.assert_almost_equal(
        sols[:, -1].std(axis=0), (theoretical_variance(N * h)) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(axis=0), theoretical_mean(N * h), decimal=1)
    npt.assert_almost_equal(
        sols[:, 1].mean(axis=0), theoretical_mean(h), decimal=2
    )
    npt.assert_array_almost_equal(
        sols[:, 1].var(),
        theoretical_variance(h),
        decimal=2,
    )


def test_ibm_path():
    JAX_KEY = jax.random.PRNGKey(1337)
    key = jax.random.split(JAX_KEY, 1)
    x0, drift, sigma, _ = ibm()
    N = 100
    dt = 1.0 / N

    incs = jax.random.normal(key, shape=(N, 1))
    linspace, sols = euler_maruyama_pathwise(incs, init=x0, drift=drift, sigma=sigma, h=dt, N=N, standard=True)
    normalized_incs = incs * dt ** 0.5

    # Reconstruct the IBM path.
    B = jnp.cumsum(normalized_incs)
    B = jnp.insert(B, 0, 0.0)
    V = x0[1] + B
    intB = jnp.cumsum(B) * dt
    U = x0[0] + x0[1] * linspace[1:] + intB[:-1]
    U = jnp.insert(U, 0, x0[0])

    npt.assert_array_almost_equal(sols[:, 1], V, decimal=5)
    npt.assert_array_almost_equal(sols[:, 0], U, decimal=5)
