import jax
import jax.numpy as jnp


def solver(key, init, drift, sigma, h, N):
    # simulate hypoelliptic diffusion with 1.5 scheme
    # https://arxiv.org/abs/1707.04235
    # does not support non diagonal diffusion matrix or maybe it does..
    # (maybe not taking into account non diagonal diffusion in leading order var.. ?)
    # up to the leading order (eq . 32)

    dim = drift(init).shape[0]

    a = lambda x: drift(x)[0]
    Gamma = lambda x: sigma(x)[1:, :]
    GammaGammaT = lambda x: Gamma(x) @ Gamma(x).T
    partialua = lambda x: jax.jacfwd(a)(x)[1:]
    partialuvdrift = lambda x: jax.jacfwd(drift)(x)

    def laplaceweighted(sig, f, x):
        # general case
        H = jax.hessian(f)(x)
        sigmasq = sig(x) @ sig(x).T
        return jax.vmap(lambda h: jnp.einsum('ij,ij->', sigmasq, h))(H)
        #special case for diagonal diffusion matrix
        #diagH = jax.vmap(jnp.diag)(jax.hessian(f)(x))
        #diagsigma = jnp.diag(sig(x) @ sig(x).T)
        #return jax.vmap(lambda diagh: jnp.dot(diagsigma, diagh.T))(diagH)

    def body(x, key):
        key_k = key
        bm_key = jax.random.split(key_k, 1)
        epsilon_k = jax.random.multivariate_normal(bm_key, jnp.zeros(dim),
                                                   jnp.block([[jnp.reshape(partialua(x) @ GammaGammaT(x) @ partialua(x).T * h ** 3 / 3, (1, 1)), jnp.reshape(partialua(x) @ GammaGammaT(x) * h ** 2 / 2, (1, dim - 1))],
                                                              [jnp.reshape(partialua(x) @ GammaGammaT(x) * h ** 2 / 2, (dim - 1 , 1)), GammaGammaT(x) * h]])
                                                   ) # up to leading order...
        out = x + h * drift(x) + h ** 2 / 2 * partialuvdrift(x) @ drift(x) + h ** 2 / 4 * laplaceweighted(sigma, drift, x) + epsilon_k
        return out, out

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * h, N + 1)
    inps = keys
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples

def exact_solver(key, init, drift, sigma, h, N):
    # simulate hypoelliptic diffusion with 1.5 scheme
    # https://arxiv.org/abs/1707.04235
    # does not support non diagonal diffusion matrix
    # up to the leading order (eq . 32)

    dim = drift(init).shape[0]

    A = lambda x: drift(x)[1:]
    Gamma = lambda x: sigma(x)[1:, :]
    partialuvdrift = lambda x: jax.jacfwd(drift)(x)
    partialudrift = lambda x: partialuvdrift(x)[:, 1:]
    partialuA = lambda x: partialudrift(x)[1:, :]
    diagGamma = lambda x: jnp.diag(Gamma(x))
    partialuGamma_diag = lambda x: jax.jacfwd(lambda z: jnp.diag(Gamma(z)))(x)[:, 1:]
    def laplaceweighted(sig, f, x):
        # special case for diagonal diffusion matrix
        diagH = jax.vmap(jnp.diag)(jax.hessian(f)(x))
        diagsigma = jnp.diag(sig(x) @ sig(x).T)
        return jax.vmap(lambda diagh: jnp.dot(diagsigma, diagh.T))(diagH)

    def body(x, key):
        key_k = key
        bm_key = jax.random.split(key_k, 1)
        zeta_eta_k = jax.random.multivariate_normal(bm_key, jnp.zeros(2 * dim - 2), jnp.block([[jnp.eye(dim - 1) * h ** 3 / 3 , jnp.eye(dim - 1) * h ** 2 / 2],
                                                                                       [jnp.eye(dim - 1) * h ** 2 / 2, jnp.eye(dim - 1) * h]]))
        zeta_k = zeta_eta_k[:dim-1]
        eta_k = zeta_eta_k[dim-1:]
        first_term = partialudrift(x) @ Gamma(x) @ zeta_k #partial_u(a, A) Gamma zeta
        second_term_U = Gamma(x) @ eta_k + partialuA(x) @ Gamma(x) @ zeta_k + \
                      0.5 * partialuGamma_diag(x)  @ Gamma(x) @ (jnp.square(eta_k) - jnp.ones((dim - 1, )) * h) + \
                      partialuGamma_diag(x) @ A(x) @ (h * eta_k - zeta_k) + 0.5 * laplaceweighted(sigma, diagGamma, x) @ (h * eta_k - zeta_k) + \
                      0.5 * (partialuGamma_diag(x)  @ partialuGamma_diag(x) @ Gamma(x) + laplaceweighted(sigma, diagGamma, x) @ (1 / 3 * eta_k @ eta_k.T - h * jnp.ones((dim - 1, )))) @ eta_k
        out = x + h * drift(x) + h ** 2 / 2 * partialuvdrift(x) @ drift(x) + \
              h ** 2 / 4 * laplaceweighted(sigma, drift, x) + first_term + jnp.insert(second_term_U, 0, 0)
        return out, out

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * h, N + 1)
    inps = keys
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples