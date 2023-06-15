import jax.numpy as jnp
import jax.random
import numpy.testing as npt

from bayesian_ode_solver.sde_solver import parabola_sde_solver_euler

def test_gbm():
    a = 1.0
    b = 1.0
    x0 = 1.0
    N = 1000
    T = 1
    delta = T / N

    drift = lambda x, t: a*x
    sigma = lambda x, t: b*x

    seed = jax.random.PRNGKey(1337)
    keys = jax.random.split(seed, 1_000)

    @jax.vmap
    def wrapped_parabola(key_op):
        return parabola_sde_solver_euler(key_op, drift, sigma, x0, delta, N)


    linspaces, sols = wrapped_parabola(keys)
    npt.assert_almost_equal(sols[:, -1].var(), x0 ** 2 * jnp.exp(2*(a + b**2 / 2) * T) * (jnp.exp(b * T) - 1), decimal=2)
    npt.assert_almost_equal(sols[:, -1].mean(),
                            x0 * jnp.exp((a + b**2 / 2) * T),
                            decimal=2)
