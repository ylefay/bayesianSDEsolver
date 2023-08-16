import jax
import jax.numpy as jnp


def solver(key, init, drift, sigma, h, N):
    """
    Strong Taylor scheme of order 1.5 for hypoelliptic diffusion.
    See https://arxiv.org/abs/1707.04235.
    Support only squared diagonal diffusion matrix.
    Up to the leading order (eq . 32).
    """

    dim = drift(init).shape[0]

    def a(x):
        return drift(x)[0]

    def Gamma(x):
        return sigma(x)[1:, :]

    def GammaGammaT(x):
        return Gamma(x) @ Gamma(x).T

    def partialua(x):
        return jax.jacfwd(a)(x)[1:]

    def partialuvdrift(x):
        return jax.jacfwd(drift)(x)

    def laplaceweighted(sig, f, x):
        # general case
        H = jax.hessian(f)(x)
        sigmasq = sig(x) @ sig(x).T
        return jax.vmap(lambda _h: jnp.einsum("ij,ij->", sigmasq, _h))(H)
        # special case for diagonal diffusion matrix
        # diagH = jax.vmap(jnp.diag)(jax.hessian(f)(x))
        # diagsigma = jnp.diag(sig(x) @ sig(x).T)
        # return jax.vmap(lambda diagh: jnp.dot(diagsigma, diagh.T))(diagH)

    def body(x, key):
        key_k = key
        bm_key = jax.random.split(key_k, 1)
        epsilon_k = jax.random.multivariate_normal(
            bm_key,
            jnp.zeros(dim),
            jnp.block(
                [
                    [
                        jnp.reshape(
                            partialua(x) @ GammaGammaT(x) @ partialua(x).T * h ** 3 / 3,
                            (1, 1),
                        ),
                        jnp.reshape(
                            partialua(x) @ GammaGammaT(x) * h ** 2 / 2, (1, dim - 1)
                        ),
                    ],
                    [
                        jnp.reshape(
                            partialua(x) @ GammaGammaT(x) * h ** 2 / 2, (dim - 1, 1)
                        ),
                        GammaGammaT(x) * h,
                    ],
                ]
            ),
        )  # up to leading order...
        out = (
                x
                + h * drift(x)
                + h ** 2 / 2 * partialuvdrift(x) @ drift(x)
                + h ** 2 / 4 * laplaceweighted(sigma, drift, x)
                + epsilon_k
        )
        return out, out

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * h, N + 1)
    inps = keys
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples
