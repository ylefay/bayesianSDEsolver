from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx
from bayesian_sde_solver.foster_polynomial import get_approx_fine as _get_approx_fine
from bayesian_sde_solver.ode_solvers import ekf0
from bayesian_sde_solver.sde_solvers import euler_maruyama_pathwise

fine_N = 1000
get_approx_fine = partial(_get_approx_fine, N=fine_N)
_, eval_parabola_0 = get_approx(dim=1)
parabolas, eval_parabola = get_approx_fine(dim=1)
JAX_KEY = jax.random.PRNGKey(1337)
solver = ekf0
delta = 0.1


def wrapped(_key, init, vector_field, T, N=1):
    return solver(None, init=init, vector_field=vector_field, h=T/N, N=N)


# Integrated Brownian motion
init_x0 = jnp.ones((2,))


def drift(x, t):
    return jnp.array([[0.0, 1.0], [0.0, 0.0]]) @ x


def sigma(x, t):
    return jnp.array([[0.0], [1.0]])


# GBM
init_x0 = jnp.ones((1,))


def drift(x, t):
    return jnp.exp(x)


def sigma(x, t):
    return jnp.array([[1.0]])


def customized_solver(h, M, coeffs):
    def body(x, inp):
        t_k, coeff = inp
        func = lambda t: eval_parabola_0(t, delta, *coeff)
        vector_field = lambda x, t: drift(x, t + t_k) + sigma(x, t + t_k) @ jax.jacfwd(func)(t)
        next_x = wrapped(None, init=x, vector_field=vector_field, T=h)
        # next_x = euler(init=x, vector_field=vector_field, h=h/10000, N=10000)
        return next_x, next_x

    ts = jnp.linspace(0, M * h, M + 1)
    inps = ts[:-1], coeffs
    _, samples = jax.lax.scan(body, init_x0, inps)
    samples = jnp.insert(samples, 0, init_x0, axis=0)
    return ts, samples


def experiment(delta, N):
    keys = jax.random.split(JAX_KEY, 1000)
    fine_delta = delta / fine_N
    @jax.vmap
    def local_step(key_OP):
        key = jax.random.split(key_OP, 1)[0]
        coeffs = parabolas(key,
                           delta)  # generate needed coefficients for Nfine parabolas, and the corresponding greater parabola

        fine_incs = coeffs[2]  # those are the Nfine increments
        fine_incs = fine_incs.reshape((fine_N, 1))
        BM = jnp.cumsum(fine_incs, axis=0) # computing the corresponding BM

        fine_incs *= jnp.sqrt(1 / fine_delta)  # normalizing them to var. 1, needed for euler_maruyama_pathwise
        _, X_DELTA_EULER = euler_maruyama_pathwise(fine_incs, init=init_x0, drift=drift, sigma=sigma, h=fine_delta,
                                                   N=fine_N)  # solving using EM scheme this path
        X_DELTA_EULER = X_DELTA_EULER[-1]  # taking last value, corresponding to X^{eu}_{delta} with h = fine_delta


        # in case we know the closed formula:
        def closed_formula(b, t):
            return init_x0 * jnp.exp(t/2+b)
        def closed_formula(b, t):
            return jnp.exp(t) * (init_x0 + b)
        # X_DELTA_FINE_CLOSED_FORMULA = closed_formula(BM[-1], delta)

        # solving using a chosen ODE solver (see def of customized_solver), the Nfine ODE => less than order 1.0
        # obtain 0.5.., both using EKF0 or euler
        # _, X_DELTA_FINE_EKF0_PARABOLAS = customized_solver(fine_delta, fine_N, (coeffs[2], coeffs[3])) #0.5, euler or ekf0 for each parabola
        # X_DELTA_FINE_EKF0_PARABOLAS = X_DELTA_FINE_EKF0_PARABOLAS[-1]

        # solve one step of EKF0 for the greater parabola ODE.
        func = lambda t: eval_parabola(t, delta, *coeffs)
        vector_field = lambda x, t: drift(x, t) + sigma(x, t) @ jax.jacfwd(func)(t)
        X_DELTA_EKF0_PARABOLAS = wrapped(None, init_x0, vector_field, delta, N=N)

        # in case we know the ode closed formula:
        def closed_formula_ode(w, i, t):
            return jnp.exp(t) * (init_x0 + w/delta * t + t/delta * (t/delta -1) * jnp.sqrt(6) * i)
        def closed_formula_ode(w, i, t):
            return jnp.exp(t / 2 + w / delta * t + t / delta * (t / delta - 1) * jnp.sqrt(6) * i)

        # X_DELTA_FINE_CLOSED_ODE_FORMULA = closed_formula_ode(coeffs[0], coeffs[1], delta)


        # solving the greater parabola ODE using euler with chosen number of steps => less than order 1.0
        # X_DELTA_EULER_PARABOLAS = euler(init_x0, vector_field, h=delta/10, N=10) #0.5

        return X_DELTA_EKF0_PARABOLAS, X_DELTA_EULER

    return local_step(keys)


deltas = [0.1, 0.05, 0.025, 0.01]
for delta in deltas:
    jnp.save(f'experiment_{delta}', experiment(delta, int(1/delta**0)))
