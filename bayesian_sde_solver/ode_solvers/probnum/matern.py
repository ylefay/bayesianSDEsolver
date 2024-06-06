from typing import Tuple
from math import comb
import jax.numpy as jnp
from numpy.typing import ArrayLike
from bayesian_sde_solver.ode_solvers.probnum.transition_function import \
    transition_function as compute_transition_function


def transition_function(k: int, magnitude: float, length: float, dt: float, dim: int) -> Tuple[ArrayLike, ArrayLike,
ArrayLike]:
    r"""
    Closed formula for (k+1/2)-Matérn transition function.
    Assuming same prior for each coordinate.
    Ref: Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing: A Look at Gaussian Process Regression Through Kalman Filtering (Sarkka, 2018),
    The Matérn Model: A Journey through Statistics, Numerical Analysis and Machine Learning, (Porcu, 2023).
    There is a rescaling between the canonical parameters of the covariance function:
        \lambda = \sqrt{2\nu}/l
    When k = 3/2,
        The linear SDE is given by the following system:
            dX = F X dt + L dW,
        where
            L = \begin{pmatrix}
                    0\\
                    \eta
                \end{pmatrix}
            F = \begin{pmatrix}
                    0 & 1\\
                    -\lambda^2 & -2\lambda
                \end{pmatrix}
        Thus, a closed formula for A = \exp(F * h) is given by
            A = \begin{pmatrix}
                    e^{- h \lambda} (1 + h \lambda) & e^{- h \lambda} h\\
                    -e^{-h \lambda} h \lambda^2 & e^{-h \lambda} (1 - h \lambda)
                \end{pmatrix},
        and for Q = \int_0^h \exp(F * s) L L^{\top} \exp(F^{\top} * s) \mathrm{d}s is given by
            Q =
        \begin{pmatrix}
            \frac{\eta ^2 \left(e^{-2 h \lambda } (-2 h \lambda  (h \lambda +1)-1)+1\right)}{4 \lambda ^3} & \frac{1}{2} \eta ^2 h^2 e^{-2 h \lambda } \\
            \frac{1}{2} \eta ^2 h^2 e^{-2 h \lambda } & \frac{\eta ^2 e^{-2 h \lambda } \left(-2 h \lambda  (h \lambda -1)+e^{2 h \lambda }-1\right)}{4 \lambda } \\
        \end{pmatrix}
    Otherwise, we use numerical integration to compute Q and Matrix Exponential to compute A.
    """
    nu = k + 0.5
    lambda_param = jnp.sqrt(2 * nu) / length
    if nu == 3 / 2:
        A = jnp.array([[jnp.exp(-dt * lambda_param) * (1 + dt * lambda_param), jnp.exp(- dt * lambda_param) * dt],
                       [-jnp.exp(-dt * lambda_param) * dt * lambda_param ** 2,
                        jnp.exp(-dt * lambda_param) * (1 - lambda_param * dt)]])
        A = jnp.kron(jnp.eye(dim), A)

        Q = magnitude ** 2 * jnp.array([[(1 + jnp.exp(-2 * dt * lambda_param) * (
                -1. - 2 * dt * lambda_param * (1 + dt * lambda_param))) / (4 * lambda_param ** 3),
                                         0.5 * jnp.exp(-2 * dt * lambda_param) * dt ** 2],
                                        [0.5 * jnp.exp(-2 * dt * lambda_param) * dt ** 2,
                                         jnp.exp(-2 * dt * lambda_param) * (
                                                 -1 + jnp.exp(2 * dt * lambda_param) - 2 * dt * lambda_param * (
                                                 -1 + dt * lambda_param)) / (4 * lambda_param)]])
        Q = jnp.kron(jnp.eye(dim), Q)
        m = jnp.zeros(dim * 2)
    else:
        F = jnp.block(
            [[jnp.zeros(k).reshape((k, 1)), jnp.diag(jnp.ones(k))],
             [jnp.array([-comb(k + 1, i) * lambda_param ** (-i + k + 1) for i in range(k + 1)]).reshape((1, k + 1))]]
        )
        L = jnp.append(jnp.array([0. for _ in range(k)]), jnp.array([magnitude])).reshape(k + 1, 1)
        m, Q, A = compute_transition_function(F, jnp.zeros(k + 1), L, dt)
        A = jnp.kron(jnp.eye(dim), A)
        Q = jnp.kron(jnp.eye(dim), Q)
        m = jnp.kron(jnp.ones(dim), m)

    return m, Q, A
