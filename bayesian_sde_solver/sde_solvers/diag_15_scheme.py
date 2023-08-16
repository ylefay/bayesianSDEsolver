import jax
import jax.numpy as jnp


def solver(key, init, drift, sigma, h, N):
    """
    1.5 strong order Itô-Taylor scheme.
    See Kloeden, Patten 1994 10.4 and Samson, Ditlevsen: https://arxiv.org/abs/1707.04235.
    Support only squared diagonal diffusion matrix.
    """

    dim = drift(init).shape[0]
    assert dim == sigma(init).shape[1]

    def partialxdrift(x):
        return jax.jacfwd(drift)(x)

    def diagSigma(x):
        return jnp.diag(sigma(x))

    def partialxsigma(x):
        return jax.jacfwd(diagSigma)(x)

    def laplaceweighted(sig, f, x):
        # special case for diagonal diffusion matrix
        diagH = jax.vmap(jnp.diag)(jax.hessian(f)(x))
        diagsigma = jnp.diag(sig(x) @ sig(x).T)
        return jax.vmap(lambda diagh: jnp.dot(diagsigma, diagh.T))(diagH)

    def laplacesigma(x):
        return laplaceweighted(sigma, diagSigma, x)

    def laplacedrift(x):
        return laplaceweighted(sigma, drift, x)

    def body(x, key):
        key_k = key
        bm_key = jax.random.split(key_k, 1)
        zeta_eta_k = jax.random.multivariate_normal(
            bm_key,
            jnp.zeros(2 * dim),
            jnp.block(
                [
                    [jnp.eye(dim) * h ** 3 / 3, jnp.eye(dim) * h ** 2 / 2],
                    [jnp.eye(dim) * h ** 2 / 2, jnp.eye(dim) * h],
                ]
            ),
        )
        zeta_k = zeta_eta_k[:dim]
        eta_k = zeta_eta_k[dim:]
        out = (
                x
                + h * drift(x)
                + h ** 2 / 2 * partialxdrift(x) @ drift(x)
                + h ** 2 / 4 * laplacedrift(x)
                + sigma(x) @ eta_k
                + partialxdrift(x) @ sigma(x) @ zeta_k
                + 0.5
                * partialxsigma(x)
                @ sigma(x)
                @ (jnp.square(eta_k) - h * jnp.ones(dim))
                + partialxsigma(x) @ drift(x) @ (h * eta_k - zeta_k)
                + 0.5 * laplacesigma(x) @ (h * eta_k - zeta_k)
                + 0.5
                * (partialxsigma(x) @ partialxsigma(x) @ sigma(x) + laplacesigma(x))
                @ (1 / 3 * eta_k @ eta_k.T - h * jnp.ones(dim))
        )
        return out, out

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * h, N + 1)
    inps = keys
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples


def solver_commutativenoise(key, init, drift, sigma, h, N):
    raise NotImplementedError
    # assume commutative noise of the second kind
    # support time dependent drift and diffusion
    # see Kloeden, Patten 1994 4.15
