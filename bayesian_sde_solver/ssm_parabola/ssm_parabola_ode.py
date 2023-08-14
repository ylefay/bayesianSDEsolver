from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver._utils import insert
from bayesian_sde_solver.ode_solvers.probnum import multiple_interlace


def sde_solver(
        key,
        drift: Callable,
        sigma: Callable,
        x0: ArrayLike,
        bm: Callable,
        delta: float,
        N: int,
        solver: callable
) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
    init = x0
    dim = x0.shape[0]
    get_coeffs, _ = bm(dim=dim) #foster's coefficients

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        brownian_increments, second_coeff_pol, *_ = coeffs_k
        init = multiple_interlace((x, brownian_increments, second_coeff_pol))
        next_x = solver(sample_key, init=init, delta=delta, drift=drift, diffusion=sigma, T=delta)
        return next_x, (next_x, *coeffs_k)

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj, *coeffs = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj, *coeffs

