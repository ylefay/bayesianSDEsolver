import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg


def tria(A):
    nx, ny = A.shape
    r, = jlinalg.qr(A.T, mode="r")
    n = min(nx, ny)
    return jnp.triu(r[:n, :n]).T


def predict(x, A, Q_or_cholQ, sqrt=False):
    m, P_or_cholP = x
    if sqrt:
        P_or_cholP = tria(jnp.concatenate([A @ P_or_cholP, Q_or_cholQ], axis=1))
        return A @ m, P_or_cholP

    return A @ m, A @ P_or_cholP @ A.T + Q_or_cholQ


def update(x, c, H, R_or_cholR, sqrt=False):
    m, P_or_cholP = x
    if sqrt:
        ny = c.shape[0]
        nx = m.shape[0]
        M = jnp.block([[H @ P_or_cholP, R_or_cholR],
                       [P_or_cholP, jnp.zeros_like(P_or_cholP, shape=(nx, ny))]])
        chol_S = tria(M)

        cholP = chol_S[ny:, ny:]

        G = chol_S[ny:, :ny]
        I = chol_S[:ny, :ny]

        m = m - G @ jlinalg.solve_triangular(I, c, lower=True)
        return m, cholP
    S = H @ P_or_cholP @ H.T + R_or_cholR
    S_invH = jlinalg.solve(S, H, assume_a='pos')
    K = (S_invH @ P_or_cholP).T
    b = m - K @ c
    C = P_or_cholP - K @ H @ P_or_cholP
    return b, C


def ekf(init, observation_function, A, Q_or_cholQ, R_or_cholR, params=None, sqrt=False):
    # sqrt : lower sqrt only, such as LDL^T or Cholesky
    def body(x, param):
        x = predict(x, A, Q_or_cholQ, sqrt)
        m, _ = x
        y = (m,) if param is None else (m, *param)
        H = jax.jacfwd(observation_function, 0)(*y)
        c = observation_function(*y)
        x = update(x, c, H, R_or_cholR, sqrt)
        return x, None

    traj, _ = jax.lax.scan(body, init, params)
    return traj
