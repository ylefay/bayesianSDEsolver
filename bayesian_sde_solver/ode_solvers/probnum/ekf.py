import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg


def _householder(a):
    if a.dtype == jnp.float64:
        eps = 1e-9
    else:
        eps = 1e-7

    alpha = a[0]
    s = jnp.sum(a[1:] ** 2)
    cond = s < eps

    def if_not_cond(v):
        t = (alpha ** 2 + s) ** 0.5
        v0 = jax.lax.cond(alpha <= 0, lambda _: alpha - t, lambda _: -s / (alpha + t), None)
        tau = 2 * v0 ** 2 / (s + v0 ** 2)
        v = v / v0
        v = v.at[0].set(1.)
        return v, tau

    return jax.lax.cond(cond, lambda v: (v, 0.), if_not_cond, a)


def qr_jvp_rule(primals, tangents):
    x, = primals
    dx, = tangents
    q, r = _qr(x, True)
    m, n = x.shape
    min_ = min(m, n)
    if m < n:
        dx = dx[:, :m]
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(q.T, dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - qt_dx_rinv_lower.T  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    do = do + jnp.eye(min_, dtype=do.dtype) * (qt_dx_rinv - jnp.real(qt_dx_rinv))
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return r, dr


@jax.custom_jvp
def qr(A: jnp.ndarray):
    """The JAX provided implementation is not parallelizable using VMAP. As a consequence, we have to rewrite it..."""
    return _qr(A)


qr.defjvp(qr_jvp_rule)


def _qr(A: jnp.ndarray, return_q=False):
    m, n = A.shape
    min_ = min(m, n)
    if return_q:
        Q = jnp.eye(m)

    for j in range(min_):
        # Apply Householder transformation.
        v, tau = _householder(A[j:, j])

        H = jnp.eye(m)
        H = H.at[j:, j:].add(-tau * (v[:, None] @ v[None, :]))

        A = H @ A
        if return_q:
            Q = H @ Q  # noqa

    R = jnp.triu(A[:min_, :min_])
    if return_q:
        return Q[:n].T, R  # noqa
    else:
        return R


def tria(A):
    q, r = qr(A.T)
    return r.T


def predict(x, A, Q_or_cholQ, sqrt=False):
    m, P_or_cholP = x
    if sqrt:
        P_or_cholP = tria(jnp.concatenate([A @ P_or_cholP, Q_or_cholQ], axis=1))
        return A @ m, P_or_cholP

    return A @ m, A @ P_or_cholP @ A.T + Q_or_cholQ


def update(x, c, H, R_or_cholR, sqrt=False):
    m, P_or_cholP = x
    if sqrt:
        nx = m.shape[0]
        y_diff = - H @ m - c
        ny = y_diff.shape[0]

        raise NotImplementedError
        m = m - K @ y_diff
        return m, cholP
    S = H @ P_or_cholP @ H.T + R_or_cholR
    S_invH = jlinalg.solve(S, H, sym_pos=True)
    K = (S_invH @ P_or_cholP).T
    b = m - K @ c
    C = P_or_cholP - K @ H @ P_or_cholP
    return b, C


def ekf(init, observation_function, A, Q_or_cholQ, R_or_cholR, params=None, sqrt=False):
    def body(x, param):
        m, _ = x
        y = (m,) if param is None else (m, *param)
        H = jax.jacfwd(observation_function, 0)(*y)
        c = observation_function(*y)
        x = predict(x, A, Q_or_cholQ, sqrt)
        x = update(x, c, H, R_or_cholR, sqrt)
        return x, x

    _, traj = jax.lax.scan(body, init, params)
    return traj
