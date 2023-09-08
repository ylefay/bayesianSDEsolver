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
        delta: float,
        N: int,
        solver: callable
) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
    init = x0
    """
    Similar to sde_solver in bayesian_sde_solver/sde_solver.py 
    Including the Brownian parabola coefficients as part of the state.
    The ssm_transtion function must be designed to handle the parabola coefficients (i.e based on bayesian_sde_solver.ssm_parabola.ekf._solver)
    And returns them separately.
    """

    @progress_bar_scan(num_samples=N, message=f"N={N}")
    def body(x, inp):
        _, key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        sigma_shifted = lambda z, t: sigma(z, t + t_k)
        drift_shifted = lambda z, t: drift(z, t + t_k)
        next_x, *coeffs_k = solver(sample_key, init=x, delta=delta, drift=drift_shifted,
                                           diffusion=sigma_shifted, T=delta)
        return next_x, (next_x, *coeffs_k)

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = jnp.arange(N), keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj, *coeffs = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj, *coeffs
