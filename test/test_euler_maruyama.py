import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import euler_maruyama


def test_gbm_moment():
    # this tests the moments of the EM method for the geometric brownian motion sde.

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    a = 1
    b = 1
    drift = lambda x, t: a * x
    sigma = lambda x, t: b * jnp.array([x])

    x0 = jnp.ones((1,))
    N = 100
    h = 1.0 / N

    @jax.vmap
    def wrapped_euler_maruyama(key_op):
        return euler_maruyama(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    linspaces, sols = wrapped_euler_maruyama(keys)

    npt.assert_almost_equal(sols[:, -1].std(), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1)
    npt.assert_almost_equal(sols[:, -1].mean(), x0 * jnp.exp(a), decimal=1)
    npt.assert_almost_equal(sols[:, 1].mean(), x0 * (1+a*h+(a*h)**2/2), decimal=2) #strong order 1, locally 2
    npt.assert_array_almost_equal(sols[:, 1].var(), x0**2*(1 + 2 * a * h + (2 * a * h) ** 2 / 2)*(b ** 2 * h + (b ** 2 * h) ** 2 / 2), decimal=2)