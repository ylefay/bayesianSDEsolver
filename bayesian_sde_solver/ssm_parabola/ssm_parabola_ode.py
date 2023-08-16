from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver._utils import insert


def sde_solver(
        key,
        drift: Callable,
        sigma: Callable,
        x0: ArrayLike,
        delta: float,
        N: int,
        solver: callable
) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
    init = x0

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        sigma_shifted = lambda z, t: sigma(z, t + t_k)
        drift_shifted = lambda z, t: drift(z, t + t_k)
        next_x = solver(sample_key, init=x, delta=delta, drift=drift_shifted, diffusion=sigma_shifted, T=delta)
        return next_x, next_x

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj
