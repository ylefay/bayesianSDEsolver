import jax.numpy as jnp
import numpy.testing as npt
from jax import random

from bayesian_sde_solver.ode_solvers.probnum.ekf import update, predict

key = random.PRNGKey(1337)


def test_agree_predict_sqrt():
    nx = 2
    ny = 2
    Q = random.uniform(key, (ny, 10 * ny))
    Q = Q @ Q.T
    cholQ = jnp.linalg.cholesky(Q)
    A = random.uniform(key, (ny, nx))
    P = random.uniform(key, (nx, 10 * nx))
    P = P @ P.T
    m = random.uniform(key, (nx,))
    x = (m, P)
    cholx = (m, jnp.linalg.cholesky(P))
    p_sqrt_m, _p_cholP = predict(cholx, A, cholQ, sqrt=True)
    p_m, p_P = predict(x, A, Q, sqrt=False)
    npt.assert_allclose(p_sqrt_m, p_m, rtol=10e-5)
    npt.assert_allclose(_p_cholP @ _p_cholP.T, p_P, rtol=10e-5)


def test_agree_update_sqrt():
    nx = 2
    ny = 2
    H = random.uniform(key, (ny, nx))
    c = random.uniform(key, (ny,))
    P = random.uniform(key, (nx, 10 * nx))
    P = P @ P.T
    m = random.uniform(key, (nx,))
    x = (m, P)
    cholx = (m, jnp.linalg.cholesky(P))
    R = random.uniform(key, (ny, 10 * ny))
    R = R @ R.T
    cholR = jnp.linalg.cholesky(R)
    u_sqrt_m, _u_cholP = update(cholx, c, H, cholR, sqrt=True)
    u_m, u_P = update(x, c, H, R, sqrt=False)
    npt.assert_allclose(_u_cholP @ _u_cholP.T, u_P, rtol=10e-5)
    npt.assert_allclose(u_sqrt_m, u_m, rtol=10e-5)
