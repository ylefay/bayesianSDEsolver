import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.ito_stratonovich import to_stratonovich, to_ito


def test_ito_strato_symmetry():
    # this tests that the ito to stratonovich conversion is the inverse of
    # the stratonovich to ito conversion and vice versa
    drift = lambda x, t: jnp.ones((1,))
    diff = lambda x, t: jnp.array([x])
    _drift, _diff = to_stratonovich(*to_ito(drift, diff))
    x = jnp.ones((1,)) * 0.5
    t = 1.0

    npt.assert_equal(_drift(x, t), drift(x, t))

    drift = lambda x, t: x
    diff = lambda x, t: jnp.array([[x[0], 2 * x[1]],
                                   [3 * x[0] + 4 * x[1], 5 * x[1]]])
    tilde_drift, tilde_diff = to_stratonovich(*to_ito(drift, diff))
    x = jnp.array([0.5, 0.9])
    t = 1.0

    ito_drift, _ = to_ito(drift, diff)
    npt.assert_array_almost_equal(tilde_drift(x, t), drift(x, t))


def test_ito_strato_diagonal():
    # this tests that in the case of a diagonal diffusion matrix,
    # the stratonovich drift is x - 0.5 * x * identity
    drift = lambda x, t: x
    diff = lambda x, t: jnp.diag(x)

    strat_drift, strat_diff = to_stratonovich(drift, diff)

    x = jnp.array([0.5, 0.9])
    t = 1.0
    npt.assert_array_almost_equal(strat_drift(x, t), x - 0.5 * x)


def test_ito_strato_1d():
    # same as precedent but in 1d
    drift = lambda x, t: 1.0
    diff = lambda x, t: jnp.array([x])
    x = jnp.ones((1,)) * 0.5
    t = 1.0
    strat_drift, strat_diff = to_stratonovich(drift, diff)
    npt.assert_equal(strat_drift(x, t), 1.0 - x * 0.5)


def test_ito_strato_md():
    # this tests that the correction is indeed correct,
    # see Kloeden, Pattern, 1999, chapter 4.9.
    drift = lambda x, t: x
    diff = lambda x, t: jnp.array([[x[0], 2 * x[1]],
                                   [3 * x[0] + 4 * x[1], 5 * x[1]]])
    x = jnp.array([0.5, 0.9])
    t = 1.0
    ito_drift, _ = to_ito(drift, diff)
    npt.assert_array_almost_equal(ito_drift(x, t), drift(x, t) +
                                  0.5 * jnp.array([(x[0] * 1 + 2 * x[1] * 0 + (3 * x[0] + 4 * x[1]) * 0 + 5 * x[1] * 2),
                                                   (x[0] * 3 + 2 * x[1] * 0 + (3 * x[0] + 4 * x[1]) * 4 + 5 * x[
                                                       1] * 5)]))
