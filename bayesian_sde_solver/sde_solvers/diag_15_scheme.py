import jax
import jax.numpy as jnp


def solver(key, init, drift, sigma, h, N):
    # squared diagonal diffusion matrix

    dim = drift(init).shape[0]
    assert dim == sigma(init).shape[1]
    partialxdrift = lambda x: jax.jacfwd(drift)(x)
    diagSigma = lambda x: jnp.diag(sigma(x))
    partialxsigma = lambda x: jax.jacfwd(diagSigma)(x)

    def laplaceweighted(sig, f, x):
        # special case for diagonal diffusion matrix
        diagH = jax.vmap(jnp.diag)(jax.hessian(f)(x))
        diagsigma = jnp.diag(sig(x) @ sig(x).T)
        return jax.vmap(lambda diagh: jnp.dot(diagsigma, diagh.T))(diagH)

    laplacesigma = lambda x: laplaceweighted(sigma, diagSigma, x)
    laplacedrift = lambda x: laplaceweighted(sigma, drift, x)

    def body(x, key):
        key_k = key
        bm_key = jax.random.split(key_k, 1)
        zeta_eta_k = jax.random.multivariate_normal(bm_key, jnp.zeros(2 * dim), jnp.block(
            [[jnp.eye(dim) * h ** 3 / 3, jnp.eye(dim) * h ** 2 / 2],
             [jnp.eye(dim) * h ** 2 / 2, jnp.eye(dim) * h]]))
        zeta_k = zeta_eta_k[:dim]
        eta_k = zeta_eta_k[dim:]
        out = x + h * drift(x) + h ** 2 / 2 * partialxdrift(x) @ drift(x) + h ** 2 / 4 * laplacedrift(x) + \
              sigma(x) @ eta_k + partialxdrift(x) @ sigma(x) @ zeta_k + \
              0.5 * partialxsigma(x) @ sigma(x) @ (jnp.square(eta_k) - h * jnp.ones(dim)) + \
              partialxsigma(x) @ drift(x) @ (h * eta_k - zeta_k) + \
              0.5 * laplacesigma(x) @ (h * eta_k - zeta_k) + \
              0.5 * (partialxsigma(x) @ partialxsigma(x) @ sigma(x) + laplacesigma(x)) @ (
                          1 / 3 * eta_k @ eta_k.T - h * jnp.ones(dim))
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
    n = drift(init, 0.).shape[0]
    m = sigma(init, 0.).shape[1]

    # f is a scalar function
    """"def L0(f):
        return lambda x, t: \
            jax.jacfwd(f, argnums=1)(x, t) + f(x, t) @ jax.jacfwd(f, argnums=0)(x, t) + \
            0.5 * jnp.einsum('kj,lj,kl->', sigma(x, t), sigma(x, t), jax.hessian(f, argnums=0)(x, t))
    def L(f):
        return lambda x, t: \
                    jnp.einsum('kj->j', jax.jacfwd(f, argnums=0)(x, t))

    ZW = jax.random.multivariate_normal(bm_key, jnp.zeros(2 * m), jnp.block(
        [[jnp.eye(m) * h ** 3 / 3, jnp.eye(m) * h ** 2 / 2],
         [jnp.eye(m) * h ** 2 / 2, jnp.eye(m) * h]]))
    Z = ZW[:m]
    W = ZW[m:]
    out = x + drift(x, t) * h + sigma(x, t) @ W + \
        0.5 * L0()

    pass"""