import jax
import jax.numpy as jnp
from jax import grad

from bayesian_ode_solver.ode_solvers import euler
from bayesian_ode_solver.foster_polynomial import get_approx as parabola_approx

def sde_solver(
        key,
        drift,
        sigma,
        x0,
        bm,
        delta,
        N,
        ode_int,
        batch_size=None,
):
    init = x0
    get_coeffs, eval_fn = bm()

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        func = lambda t: eval_fn(t, delta, *coeffs_k)
        vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) * grad(func)(t)
        next_x = ode_int(sample_key, init=x, vector_field=vector_field, T=delta)
        return next_x, next_x

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)

    return ts, samples


def wrapped_euler(_key, init, vector_field, T):
    # 10 points euler
    M = 100_000
    return euler(init=init, vector_field=vector_field, h=T / M, N=M)


def parabola_sde_solver_euler(key, drift, sigma, x0, delta, N):
    return sde_solver(key=key, drift=drift, sigma=sigma, x0=x0, bm=parabola_approx, delta=delta, N=N, ode_int=wrapped_euler)