import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import euler_maruyama, euler_maruyama_pathwise


def test_gbm_moment():
    # this tests the moments of the EM method for the geometric brownian motion sde.

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000_000)

    a = 1
    b = 1

    def drift(x, t):
        return a * x

    def sigma(x, t):
        return b * jnp.diag(x)

    x0 = jnp.ones((1,))
    N = 100
    h = 1.0 / N

    @jax.vmap
    def wrapped_euler_maruyama(key_op):
        return euler_maruyama(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    linspaces, sols = wrapped_euler_maruyama(keys)

    npt.assert_almost_equal(
        sols[:, -1].std(axis=0), x0 * jnp.exp(a) * (jnp.exp(b) - 1) ** 0.5, decimal=1
    )
    npt.assert_almost_equal(sols[:, -1].mean(axis=0), x0 * jnp.exp(a), decimal=1)
    npt.assert_almost_equal(
        sols[:, 1].mean(axis=0), x0 * (1 + a * h + (a * h) ** 2 / 2), decimal=2
    )  # strong order 1, locally 2
    npt.assert_array_almost_equal(
        sols[:, 1].var(),
        x0 ** 2
        * (1 + 2 * a * h + (2 * a * h) ** 2 / 2)
        * (b ** 2 * h + (b ** 2 * h) ** 2 / 2),
        decimal=2,
    )


def test_ibm_path():
    JAX_KEY = jax.random.PRNGKey(1337)
    key = jax.random.split(JAX_KEY, 1)

    def drift(x, t):
        return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x

    def sigma(x, t):
        return jnp.array([[0.0], [1.0]])

    x0 = jnp.ones((2,))
    N = 100
    dt = 1.0 / N
    incs = jax.random.normal(key, shape=(N, 1))
    linspace, sols = euler_maruyama_pathwise(incs, init=x0, drift=drift, sigma=sigma, h=dt, N=N, standard=True)
    normalized_incs = incs * dt ** 0.5
    B = jnp.cumsum(normalized_incs)
    B = jnp.insert(B, 0, 0.0)
    V = x0[1] + B
    intB = jnp.cumsum(B) * dt
    U = x0[0] + x0[1] * linspace[1:] + intB[:-1]
    U = jnp.insert(U, 0, x0[0])
    npt.assert_array_almost_equal(sols[:, 1], V, decimal=5)
    npt.assert_array_almost_equal(sols[:, 0], U, decimal=5)
