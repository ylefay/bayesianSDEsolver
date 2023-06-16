import jax
import jax.numpy as jnp
from jax import grad


def sde_solver(
        key,
        drift,
        sigma,
        x0,
        bm,
        delta,
        N,
        ode_int,
):
    init = x0
    get_coeffs, eval_fn = bm()
    dim = sigma(x0, 0.).shape[1]

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta, dim)
        func = lambda t: eval_fn(t, delta, *coeffs_k)
        vector_field = lambda z, t: drift(z, t + t_k) + jnp.dot(sigma(z, t + t_k), jax.jacfwd(func)(t))
        next_x = ode_int(sample_key, init=x, vector_field=vector_field, T=delta)
        return next_x, next_x

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples[:, 0]

