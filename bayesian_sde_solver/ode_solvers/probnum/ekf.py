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


def update(x, c, H, R_or_cholR, lower_sqrt=False):
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
        return m, cholP
    S = H @ P_or_cholP @ H.T + R_or_cholR
    S_invH = linalg.solve(S, H, assume_a='pos')
    K = (S_invH @ P_or_cholP).T
    b = m - K @ c
    C = P_or_cholP - K @ H @ P_or_cholP
    return b, C


def ekf(init, observation_function, A, Q_or_cholQ, xi, R_or_cholR, params=None, lower_sqrt=False):
    """
    Extended Kalman filter with optional lower square root implementation, such as Cholesky decomposition.
    See https://arxiv.org/abs/2207.00426.
    """

    def body(x, param):
        x = predict(x, A, Q_or_cholQ, xi, lower_sqrt)
        m, _ = x
        y = (m,) if param is None else (m, *param)
        H = jax.jacfwd(observation_function, 0)(*y)
        c = observation_function(*y)
        x = update(x, c, H, R_or_cholR, lower_sqrt)
        return x, None

    traj, _ = jax.lax.scan(body, init, params)
    return traj
