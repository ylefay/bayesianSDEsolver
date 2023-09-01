from functools import partial
from bayesian_sde_solver.utils import progress_bar_scan

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich, to_ito
from bayesian_sde_solver.ode_solvers import ekf0_2, ekf1_2, ekf0
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function

JAX_KEY = jax.random.PRNGKey(1337)

_solver = ekf0
theta = 1.0


def fhn():
    gamma = 1.5
    sig = 0.3
    eps = 0.1
    alpha = 0.8
    s = 0.0
    x0 = jnp.zeros((2,))

    def drift(x, t):
        return jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
            [s / eps - x[0] ** 3 / eps, alpha])

    def sigma(x, t):
        return jnp.array([[0.0], [sig]])

    return x0, drift, sigma


x0, drift, sigma = fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)
init = x0
if _solver in [ekf0_2, ekf1_2]:
    P0 = jnp.zeros((x0.shape[0], x0.shape[0]))
    init = (x0, P0)


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s)", excluded=(1, 2, 3,))
def experiment(delta, N, M, fine):
    # special sde_solver function to solve RAM issue
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
            bm: Callable,
            delta: float,
            N: int,
            ode_int: Callable,
    ) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
        """
        Same as sde_solver but with an intermediary fine Euler Maruyama scheme.
        Used for RAM expensive pathwise comparisons.
        """
        init = x0
        get_coeffs, eval_fn = bm()

        @progress_bar_scan(num_samples=N, message=f"N={N}")
        def body(x, inp):
            x1, x2 = x
            _, key_k, t_k = inp
            bm_key, sample_key = jax.random.split(key_k, 2)
            coeffs_k = get_coeffs(bm_key, delta)
            func = lambda t: eval_fn(t, delta, *coeffs_k)
            drift_shifted = lambda z, t: drift(z, t + t_k)
            sigma_shifted = lambda z, t: sigma(z, t + t_k)

            vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) @ jax.jacfwd(func)(t)
            next_x = ode_int(sample_key, init=x1, vector_field=vector_field, T=delta)
            dt = delta / fine
            standard_incs = coeffs_k[2] * jnp.sqrt(1 / dt)
            drift_shifted_ito, sigma_shifted_ito = to_ito(drift_shifted, sigma_shifted)
            _, euler_path = euler_maruyama_pathwise(standard_incs, init=x2, drift=drift_shifted_ito,
                                                    sigma=sigma_shifted_ito,
                                                    h=dt, N=fine)
            next_x2 = euler_path[-1]
            return (next_x, next_x2), (next_x, next_x2)

        keys = jax.random.split(key, N)
        ts = jnp.linspace(0, N * delta, N + 1)

        inps = jnp.arange(N), keys, ts[:-1]
        _, samples = jax.lax.scan(body, (init, init), inps)
        traj, traj2 = samples
        traj = insert(traj, 0, init, axis=0)
        traj2 = insert(traj2, 0, init, axis=0)
        return ts, traj, traj2

    keys = jax.random.split(JAX_KEY, 1_000)

    prior = IOUP_transition_function(theta=theta, sigma=1.0, dt=delta / M, q=1, dim=x0.shape[0])
    solver = partial(_solver, prior=prior, noise=None)

    def wrapped(_key, init, vector_field, T):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M)

    get_approx_fine = partial(_get_approx_fine, N=fine)

    @jax.vmap
    def wrapped_filter_parabola(key_op):
        return sde_solver(
            key=key_op,
            drift=drift_s,
            sigma=sigma_s,
            x0=init,
            bm=get_approx_fine,
            delta=delta,
            N=N,
            ode_int=wrapped,
        )

    linspaces, sols, sol2 = wrapped_filter_parabola(keys)
    if solver in [ekf0_2, ekf1_2]:
        sols = sols[0]

    return sols, sol2


Ns = jnp.array([4, 8, 16, 32, 64])
deltas = jnp.array([0.25, 0.125, 0.0625, 0.03125, 0.015625])
fineDeltas = Ns ** 1
Mdeltas = jnp.ones((len(deltas),)) * Ns ** 0
Ndeltas = Ns

folder = "./"
solver_name = "ekf0"
problem_name = "test"
prefix = f"{solver_name}_{problem_name}"
for n in range(len(Ndeltas)):
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    fine = int(fineDeltas[n])
    s1, s2 = experiment(delta, N, M, fine)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{M}', s1)
    jnp.save(f'{folder}/{prefix}_pathwise_sols2_{N}_{fine}', s2)
