import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg


def tria(A):
    nx, ny = A.shape
    r, = linalg.qr(A.T, mode="r")
    n = min(nx, ny)
    return jnp.triu(r[:n, :n]).T


def predict(x, A, Q_or_cholQ, xi, lower_sqrt=False):
    m, P_or_cholP = x
    if lower_sqrt:
        P_or_cholP = tria(jnp.concatenate([A @ P_or_cholP, Q_or_cholQ], axis=1))
        return A @ m + xi, P_or_cholP

    return A @ m + xi, A @ P_or_cholP @ A.T + Q_or_cholQ


def update(x, c, H, R_or_cholR, lower_sqrt=False, return_S=False):
    m, P_or_cholP = x
    if lower_sqrt:
        ny = c.shape[0]
        nx = m.shape[0]
        M = jnp.block([[H @ P_or_cholP, R_or_cholR],
                       [P_or_cholP, jnp.zeros_like(P_or_cholP, shape=(nx, ny))]])
        chol_S = tria(M)

        cholP = chol_S[ny:, ny:]

        G = chol_S[ny:, :ny]
        I = chol_S[:ny, :ny]

        m = m - G @ linalg.solve_triangular(I, c, lower=True)
        return (m, cholP), I if return_S else None
    S = H @ P_or_cholP @ H.T + R_or_cholR
    S_invH = linalg.solve(S, H, assume_a='pos')
    K = (S_invH @ P_or_cholP).T
    b = m - K @ c
    C = P_or_cholP - K @ H @ P_or_cholP
    return (b, C), S if return_S else None


def ekf(init, observation_function, A, Q_or_cholQ, xi, R_or_cholR, params=None, lower_sqrt=False,
        return_UC=False):
    """
    Extended Kalman filter with optional lower square root implementation, such as Cholesky decomposition.
    See https://arxiv.org/abs/2207.00426.
    """

    def body_return_UC(inps, param):
        """
        Handle the return of the error measurement mean and variance.
        Needed for Uncertainty Calibration.
        """
        x, *_ = inps
        x = predict(x, A, Q_or_cholQ, xi, lower_sqrt)
        m, _ = x
        y = (m,) if param is None else (m, *param)
        H = jax.jacfwd(observation_function, 0)(*y)
        c = observation_function(*y)
        x, S_or_cholS = update(x, c, H, R_or_cholR, lower_sqrt,
                               return_S=True)
        if lower_sqrt:
            S_or_cholS = S_or_cholS @ S_or_cholS
        return (x, S_or_cholS, c), None

    def body(inps, param):
        x = inps
        x = predict(x, A, Q_or_cholQ, xi, lower_sqrt)
        m, _ = x
        y = (m,) if param is None else (m, *param)
        H = jax.jacfwd(observation_function, 0)(*y)
        c = observation_function(*y)
        x, _ = update(x, c, H, R_or_cholR, lower_sqrt, return_S=False)
        return x, None

    if return_UC:
        S_or_cholS_shape = R_or_cholR.shape
        z_shape = R_or_cholR.shape[0]
        init = (init, jnp.empty(S_or_cholS_shape), jnp.empty(z_shape))
        traj, _ = jax.lax.scan(body_return_UC, init, params)
    else:
        traj, _ = jax.lax.scan(body, init, params)
    return traj
