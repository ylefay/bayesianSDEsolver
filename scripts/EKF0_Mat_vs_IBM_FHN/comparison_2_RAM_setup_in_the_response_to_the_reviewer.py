from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ito_stratonovich import to_stratonovich
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function, matern_transition_function
from bayesian_sde_solver.ode_solvers.probnum import get_independently_factorized_prior, pad_prior
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise
from bayesian_sde_solver.utils import progress_bar
from bayesian_sde_solver.utils.ivp import double_ibm, ibm, fhn

JAX_KEY = jax.random.PRNGKey(1337)

order = 2
solver_name = "EKF0"
problem_name = "FHN"
prefix = f"{solver_name}_{problem_name}_review"
folder = "./"

_solver = ekf0

x0, drift, sigma, _ = ibm()
x0, drift, sigma, _ = double_ibm()
x0, drift, sigma, _, _ = fhn()
drift_s, sigma_s = to_stratonovich(drift, sigma)
init = x0


@partial(jnp.vectorize, signature="()->(d,n,s),(d,n,s),(d,n,s)", excluded=(1, 2, 3,))
def experiment(delta, N, M, fine):
    # special sde_solver function to solve RAM issue
    from typing import Callable, Tuple

    import jax
    import jax.numpy as jnp
    from numpy.typing import ArrayLike

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
        Same as sde_solver but with an intermediary fine Euler Maruyama scheme.
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
            drift_shifted = lambda z, t: drift(z, t + t_k)
            sigma_shifted = lambda z, t: sigma(z, t + t_k)

            vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) @ jax.jacfwd(func)(t)
            next_x = ode_int(sample_key, init=x1, vector_field=vector_field, T=delta)
            dt = delta / fine
            incs = coeffs_k[2]
            # assuming additive noise
            # drift_shifted_ito, sigma_shifted_ito = to_ito(drift_shifted, sigma_shifted)
            drift_shifted_ito, sigma_shifted_ito = drift_shifted, sigma_shifted
            _, euler_path = euler_maruyama_pathwise(incs, init=x2, drift=drift_shifted_ito,
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

    keys = jax.random.split(JAX_KEY, 100_000)

    prior_2_IBM = IOUP_transition_function(theta=1., sigma=1.0, dt=delta / M, q=2, dim=1)
    prior_1_IBM = IOUP_transition_function(theta=1., sigma=1.0, dt=delta / M, q=1, dim=1)
    prior_1_IBM = pad_prior(prior_1_IBM, q=2)
    prior_2_mat = matern_transition_function(k=2, magnitude=1.0, length=jnp.sqrt(5) / 3, dt=delta / M, dim=1)
    prior_1_mat = matern_transition_function(k=1, magnitude=1.0, length=jnp.sqrt(3) / 2, dt=delta / M, dim=1, mc=True)
    prior_1_mat = pad_prior(prior_1_mat, q=2)

    prior_IBM = get_independently_factorized_prior((prior_2_IBM, prior_1_IBM))
    prior_mat = get_independently_factorized_prior((prior_2_mat, prior_1_mat))
    solver = partial(_solver, noise=None, sqrt=False)

    def wrapped(_key, init, vector_field, T, prior):
        return solver(_key, init=init, vector_field=vector_field, h=T / M, N=M, prior=prior)

    get_approx_fine = partial(_get_approx_fine, N=fine, dim=sigma(x0, 0.).shape[1])

    def wrapped_filter_parabola(key_op, wrapped):
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

    linspaces, sols_ioup, sols = jax.vmap(partial(wrapped_filter_parabola, wrapped=partial(wrapped, prior=prior_IBM)))(
        keys)
    _, sols_mat, _ = jax.vmap(partial(wrapped_filter_parabola, wrapped=partial(wrapped, prior=prior_mat)))(keys)
    return sols, sols_ioup, sols_mat


deltas = 1 / jnp.array([16, 32, 64, 128, 256, 512, 1024])
# deltas = 1 / jnp.array([16, 32, 64])
Ns = 1 / deltas
fineN = Ns ** 1.0
Mdeltas = jnp.ones((len(deltas),)) * (Ns) ** 0.
T = 1.0
Ndeltas = T / deltas

print(prefix)
for n in range(len(Ndeltas)):
    delta = deltas[n]
    N = int(Ndeltas[n])
    M = int(Mdeltas[n])
    fine = int(fineN[n])
    with jax.disable_jit(True):
        sols, sols_ioup, sols_mat = experiment(delta, N, M, fine)
    print(sols_ioup)
    print(sols_mat)
    jnp.save(f'{folder}/{prefix}_pathwise_sols_{N}_{fine}', sols)
    jnp.save(f'{folder}/{prefix}_IOUP_{order}_pathwise_sols2_{N}_{M}', sols_ioup)
    jnp.save(f'{folder}/{prefix}_Matern_{order}_pathwise_sols2_{N}_{M}', sols_mat)
