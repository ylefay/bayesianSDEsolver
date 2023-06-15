import numpy as np
import numpy.testing as npt

from bayesian_ode_solver.ode_solvers import euler


def test_exponential():
    N = 1_000
    h = 1 / N
    x0 = 1.0

    def vector_field(x, t):
        return x

    x = euler(x0, vector_field, h, N)
    npt.assert_almost_equal(x, np.exp(1), decimal=3)

