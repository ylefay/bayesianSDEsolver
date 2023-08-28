import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.ode_solvers.probnum import ekf
from bayesian_sde_solver.ode_solvers.probnum import interlace


def _solver(init, drift, diffusion, delta, h, N, sqrt=True, EKF0=False, prior=None, noise=None):
    """
    EKF{0, 1} implementation for the Parabola ODE method with
    the polynomial coefficients as part of the state.
    IBM prior by default.
    One derivative of the vector field is used, q = 1.
    No observation noise by default, R = 0.
    Algorithm 4.
    """

    ts = jnp.linspace(h, N * h, N)
    dim = int(init.shape[0])
    dim_brownian = diffusion(init, 0.0).shape[1]
    assert dim == dim_brownian
    if noise is None:
        noise = jnp.zeros((dim, dim))
    else:
        assert noise.shape == (dim, dim)
        
    Evfvf = 4 / delta * diffusion(init, 0.0) @ diffusion(init, 0.0).T
    Evfw = diffusion(init, 0.0)
    Evfi = - jnp.sqrt(6) / 2 * diffusion(init, 0.0)

    mean = interlace((init, drift(init, 0.0), jnp.zeros((dim,)), jnp.zeros((dim,))))

    def block(i, j):
        return jnp.array([[0, 0, 0, 0],
                          [0, Evfvf[i, j], Evfw[i, j], Evfi[i, j]],
                          [0, Evfw[j, i], delta if i == j else 0, 0],
                          [0, Evfi[j, i], 0, delta / 2 if i == j else 0]])

    var = jnp.block([[block(i, j) for j in range(dim)] for i in range(dim)])

    def pol(t):
        H = jnp.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1 / delta, 0],
                       [0, 0, 0, jnp.sqrt(6) * (2 * t / delta - 1) / delta]])
        H = jnp.kron(jnp.eye(dim), H)
        return H

    H23 = jnp.kron(jnp.eye(dim), jnp.array([[0, 0, 1, 1]]))

    def extended_vector_field(x, t):
        return drift(x[::4], t) + diffusion(x[::4], t) @ H23 @ pol(t) @ x

    if EKF0:
        def observation_function(x, t):
            # IVP observation function
            z = x.at[::4].set(jax.lax.stop_gradient(x[::4]))
            return x[1::4] - extended_vector_field(z, t)
    else:
        def observation_function(x, t):
            # IVP observation function
            return x[1::4] - extended_vector_field(x, t)
    if prior is None:
        (
            _,
            one_block_transition_covariance,
            one_block_transition_matrix
        ) = IOUP_transition_function(theta=0.0, sigma=1.0, q=1, dt=h, dim=1)
    else:
        (
            _,
            one_block_transition_covariance,
            one_block_transition_matrix
        ) = prior

    if sqrt:
        one_block_transition_covariance = jnp.linalg.cholesky(one_block_transition_covariance)
        var = jnp.real(linalg.sqrtm(var))  # should be LDL.T, Cholesky
    init = (mean, var)

    one_block_transition_matrix = jnp.block(
        [[one_block_transition_matrix, jnp.zeros((2, 2))],
         [jnp.zeros((2, 2)), jnp.eye(2)]]
    )
    one_block_transition_covariance = jnp.block(
        [[one_block_transition_covariance, jnp.zeros((2, 2))],
         [jnp.zeros((2, 2)), jnp.zeros((2, 2))]]
    )

    transition_matrix = jnp.kron(jnp.eye(dim), one_block_transition_matrix)
    transition_covariance = jnp.kron(jnp.eye(dim), one_block_transition_covariance)

    filtered = ekf(init=init, observation_function=observation_function, A=transition_matrix,
                   Q_or_cholQ=transition_covariance, R_or_cholR=noise, params=(ts,), lower_sqrt=sqrt)

    return filtered
