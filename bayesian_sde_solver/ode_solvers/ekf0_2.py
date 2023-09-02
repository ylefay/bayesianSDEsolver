import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
from bayesian_sde_solver.ode_solvers.ekf import _solver
from bayesian_sde_solver.ode_solvers.probnum import interlace, interlace_matrix


def solver(_, init, vector_field, h, N, sqrt=False, prior=None, noise=None):
    """
    Wrapper for EKF0 with the prior being initialized at the previous posterior.
    Hence, this solver leads to one prior for the whole trajectory.
    Algorithm 3.
    """
    # Todo: check that it's correct
    m_0, P_00 = init
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
    filtered = _solver(init, vector_field, h, N, sqrt=sqrt, EKF0=True, prior=prior, noise=noise)
    m, P = filtered
    if sqrt:
        P = P @ P.T
    m_0, P_00 = m[::2], P[::2, ::2]
    return (m_0, P_00)  # return the law of X^1
