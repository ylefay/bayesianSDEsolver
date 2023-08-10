from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver._utils import insert
from bayesian_sde_solver.foster_polynomial import get_approx
from bayesian_sde_solver.ssm_parabola import ekf1_marginal_parabola
from bayesian_sde_solver.ode_solvers.probnum import multiple_interlace
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
    dim = x0.shape[0]
    get_coeffs, _ = get_approx(dim=dim)


    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        brownian_increments, levy_areas = coeffs_k
        init = multiple_interlace((x, brownian_increments, levy_areas))
        next_x = solver(sample_key, init=init, delta=delta, drift=drift, diffusion=sigma, T=delta)
        return next_x, (next_x, *coeffs_k)

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj, *coeffs = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj, *coeffs


JAX_KEY = jax.random.PRNGKey(1337)
keys = jax.random.split(JAX_KEY, 1_000_0)

def drift(x, t):
    return x


def sigma(x, t):
    return jnp.array([[1.0]])

x0 = jnp.ones((1,))


def solver(key, init, delta, drift, diffusion, T):
    h = T
    return ekf1_marginal_parabola(key, init, delta, drift, diffusion, h=h, N=1, sqrt=True)
@jax.vmap
def wrapped_filter_parabola(key_op):
    return sde_solver(key=key_op, drift=drift, sigma=sigma, x0=x0, delta=0.1, N=10, solver=solver)

_, sols, *_ = wrapped_filter_parabola(keys)
print(jnp.std(sols[:, -1]))