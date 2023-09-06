import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
from bayesian_sde_solver.ode_solvers.ekf import _solver
from bayesian_sde_solver.ode_solvers.probnum import interlace, interlace_matrix


def solver(key, init, vector_field, h, N, sqrt=False, prior=None, noise=None):
    """
    Wrapper for EKF1 with the prior being initialized at the previous posterior.
    Hence, this solver leads to one prior for the whole trajectory.
    """
    # Todo: check that it's correct
    _, m_0, P_00 = init
    dim = m_0.shape[0]
    H = jax.jacfwd(vector_field, 0)(m_0, 0.0)
    P_11 = H @ P_00 @ H.T
    P_01 = P_00 @ H.T
    P_10 = H @ P_00
    var = interlace_matrix(P_00, P_01, P_10, P_11)
    if sqrt:
        var = jnp.real(linalg.sqrtm(var))
    init = (
        interlace((m_0, vector_field(m_0, 0.0))),
        var
    )
    filtered = _solver(init, vector_field, h, N, sqrt=sqrt, EKF0=False, prior=prior, noise=noise)
    m, P = filtered
    if sqrt:
        sqrtP = P
        P = sqrtP @ sqrtP.T
    else:
        sqrtP = jnp.real(linalg.sqrtm(P))
    if key is not None:
        sample = m + sqrtP @ jax.random.multivariate_normal(key, jnp.zeros((2 * dim,)), jnp.eye(2 * dim))
    else:
        sample = m
    s_0, m_0, P_00 = sample[::2], m[::2], P[::2, ::2]
    return (s_0, m_0, P_00)  # return a sample as well as the law Y^0
