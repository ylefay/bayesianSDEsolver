from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver.utils.progress_bar import progress_bar_scan
from bayesian_sde_solver.utils.insert import insert


def sde_solver(
        key,
        drift: Callable,
        sigma: Callable,
        x0: ArrayLike,
        bm: Callable,
        delta: float,
        N: int,
        ode_int: Callable,
) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
    """
    Solve the sequence of random ODEs given a method for generating Brownian motion differentiable approximation.
    """
    init = x0
    get_coeffs, eval_fn = bm()

    @progress_bar_scan(num_samples=N, message=f"N={N}")
    def body(x, inp):
        _, key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        func = lambda t: eval_fn(t, delta, *coeffs_k)
        vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) @ jax.jacfwd(func)(t)
        next_x = ode_int(sample_key, init=x, vector_field=vector_field, T=delta)
        return next_x, (next_x, *coeffs_k)

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = jnp.arange(N), keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj, *coeffs = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj, *coeffs
