from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
from bayesian_sde_solver.utils import progress_bar
from bayesian_sde_solver.utils.ivp import random_linear_sde

JAX_KEY = jax.random.PRNGKey(1337)

solver_name = "EKF0"
problem_name = "RND_LINEAR_SDE"
prefix = f"{solver_name}_{problem_name}"
folder = "./"

_solver = ekf0


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s)", excluded=(1, 2, 3, 4))
def experiment(delta, N, M, fine, dim):
    # special sde_solver function to solve RAM issue
    from typing import Callable, Tuple

    import jax
    import jax.numpy as jnp
    from numpy.typing import ArrayLike

    from bayesian_sde_solver.utils.insert import insert

    x0, drift, sigma = random_linear_sde(key=JAX_KEY, dim=dim)
    drift_matrix = drift(jnp.identity(n=dim), 0.0)
    sigma_matrix = sigma(jnp.identity(n=dim), 0.0)
    transition_matrix = jlinalg.expm(drift_matrix * delta)
    init = x0
    fine_ts = jnp.arange(0, delta, M)
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
        Same as sde_solver but with an intermediary fine closed formula scheme,
        for linear SDE.
        Used for RAM expensive pathwise comparisons.
        """
        init = x0
        get_coeffs, eval_fn = bm()

        @progress_bar(num_samples=N, message=f"N={N}")
        def body(x, inp):
            x1, x2 = x
            _, key_k, t_k = inp
            bm_key, sample_key = jax.random.split(key_k, 2)
            coeffs_k = get_coeffs(bm_key, delta)
            func = lambda t: eval_fn(t, delta, *coeffs_k)

            vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) @ jax.jacfwd(func)(t)
            next_x = ode_int(sample_key, init=x1, vector_field=vector_field, T=delta)
            incs = coeffs_k[2]

            @jax.vmap
            def integrand(idx):
                """
                e^{M(delta-s)} @ sigma @ dW(s).
                """
                return jlinalg.expm(drift_matrix * (delta - fine_ts.at[idx].get())) @ sigma_matrix @ incs.at[idx].get()

            next_x2 = transition_matrix @ x2 + jnp.sum(integrand(idx=jnp.arange(len(incs))), axis=0)
            return (next_x, next_x2), (next_x, next_x2)

        keys = jax.random.split(key, N)
        ts = jnp.linspace(0, N * delta, N + 1)

        inps = jnp.arange(N), keys, ts[:-1]
        _, samples = jax.lax.scan(body, (init, init), inps)
        traj, traj2 = samples
        traj = insert(traj, 0, init, axis=0)
        traj2 = insert(traj2, 0, init, axis=0)
        return ts, traj, traj2

    keys = jax.random.split(JAX_KEY, 100_000)

    prior = IOUP_transition_function(theta=0., sigma=1.0, dt=delta / M, q=1, dim=x0.shape[0])
    solver = partial(_solver, prior=prior, noise=None, sqrt=True)

    def wrapped(_key, init, vector_field, T):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M)

    get_approx_fine = partial(_get_approx_fine, N=fine, dim=sigma(x0, 0.).shape[1])

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift,
            sigma=sigma,
            x0=init,
            bm=get_approx_fine,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols, sol2 = wrapped_filter_parabola(keys)
    return sols, sol2


delta = 1 / 128
N = 128
fineN = N ** 1.0
Mdelta = 1
T = 1
Ndelta = T / delta
dims = jnp.array([1, 2, 4, 8, 16, 32])

print(prefix)
for dim in dims:
    N = int(N)
    Mdelta = int(Mdelta)
    fineN = int(fineN)
    s1, s2 = experiment(delta, N, Mdelta, fineN, dim)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{Mdelta}_{dim}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fineN}_{dim}', s2)
