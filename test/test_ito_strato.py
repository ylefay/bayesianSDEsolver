from bayesian_ode_solver.ito_stratonovich import to_stratonovich, to_ito
import numpy.testing as npt
import jax.numpy as jnp

def test_ito_strato_1d():
    drift = lambda x, t: 1.0
    diff = lambda x, t: x

    _drift, _diff = to_stratonovich(*to_ito(drift, diff))

    x = 0.5
    t = 1.0

    npt.assert_equal(_drift(x, t), jnp.array([drift(x, t)]))

    drift = lambda x, t: 1.0
    diff = lambda x, t: x
    strat_drift, strat_diff = to_stratonovich(drift, diff)
    npt.assert_equal(strat_drift(x, t), jnp.array([1.0 - x * 0.5]))

def test_ito_strato_diagonal():
    drift = lambda x, t: x
    diff = lambda x, t: jnp.diag(x)

    strat_drift, strat_diff = to_stratonovich(drift, diff)

    x = jnp.array([0.5, 0.9])
    t = 1.0
    npt.assert_array_almost_equal(strat_drift(x, t), x - 0.5 * x)

def test_ito_strato_symmetry():
    drift = lambda x, t: x
    diff = lambda x, t: jnp.array([[x[0], x[1]],
                                   [x[0]+x[1], x[1]]])

    tilde_drift, tilde_diff = to_stratonovich(*to_ito(drift, diff))

    x = jnp.array([0.5, 0.9])
    t = 1.0

    ito_drift, _ = to_ito(drift, diff)
    print(ito_drift(x, t))
    npt.assert_array_almost_equal(tilde_drift(x, t), drift(x, t))
