import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from bayesian_sde_solver.ode_solvers import euler


def test_exponential():
    # this tests that the euler method for the vector field f(x) = x gives  the exponential function.
    N = 1_000
    h = 1 / N
    x0 = jnp.ones((1,))

    def vector_field(x, t):
        return x

    x = euler(x0, vector_field, h, N)
    npt.assert_almost_equal(x, np.exp(1), decimal=3)


def test_sin():
    N = 1_000
    h = 1 / N
    x0 = jnp.ones((1,))

    def vector_field(x, t):
        return jnp.sin(t)

    x = euler(x0, vector_field, h, N)
    npt.assert_almost_equal(x, 2-np.cos(1), decimal=3)
