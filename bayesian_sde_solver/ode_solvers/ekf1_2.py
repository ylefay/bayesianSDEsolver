import jax.numpy as jnp
import jax
from parsmooth import MVNStandard

from bayesian_sde_solver.ode_solvers.probnum import interlace
from bayesian_sde_solver.ode_solvers.ekf1 import _solver

def solver(key, init, vector_field, h, N):
    """
    Wrapper for EKF1 with the prior being initialized at the previous posterior.
    Hence, this solver leads to one prior for the whole trajectory.
    """
    #Todo: check that it's correct
    m_0, P_00 = init
    dim = m_0.shape[0]
    H = jax.jacfwd(vector_field, 0)(m_0, 0.0)
    P_11 = H @ P_00 @ H.T
    P_01 = P_00 @ H.T
    P_10 = H @ P_00
    var = jnp.block([[P_00, P_01],
                   [P_10,  P_11]]
        )
    _var = var.copy()
    #interlace
    _var.at[::2, ::2].set(var.at[:dim, :dim].get())
    _var.at[1::2, 1::2].set(var.at[dim:, dim:].get())
    _var.at[1::2, ::2].set(var.at[dim:, :dim].get())
    _var.at[::2, 1::2].set(var.at[:dim, dim:].get())
    init = MVNStandard(
        interlace(m_0, vector_field(m_0, 0.0)),
        _var
    )

    filtered = _solver(init, vector_field, h, N)
    m, P = filtered
    m_0, P_00 = m[::2], P[::2, ::2]
    return MVNStandard(m_0, P_00) #return the law of X^1
