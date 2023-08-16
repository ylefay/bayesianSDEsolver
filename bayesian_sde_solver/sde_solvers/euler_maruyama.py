import jax
import jax.numpy as jnp


def solver(key, init, drift, sigma, h, N):
    """
    Euler-Maruyama method.
    """
    dim = sigma(init, 0.0).shape[1]

    def body(x, inp):
        key_k, t = inp
        bm_key, _ = jax.random.split(key_k, 2)
        dW = h ** 0.5 * jax.random.normal(bm_key, shape=(dim,))
        out = x + h * drift(x, t) + sigma(x, t) @ dW
        return out, out

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * h, N + 1)
    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples


def solver_pathwise(incs, init, drift, sigma, h, N):
    """
    Euler-Maruyama method given standard increments, i.e W\sim \mathcal{N}(0_d,I_d).
    """
    def body(x, inp):
        inc, t = inp
        dW = h ** 0.5 * inc
        out = x + h * drift(x, t) + sigma(x, t) @ dW
        return out, out

    ts = jnp.linspace(0, N * h, N + 1)
    inps = incs, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples
