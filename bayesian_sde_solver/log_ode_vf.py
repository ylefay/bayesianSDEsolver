from functools import partial

import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx


@partial(jax.jit, static_argnums=(1, 2))
def lie(var, u, v):
    """
    Vector field Lie bracket.
    :param var: The Lie bracket will be evaluated at x = var.
    :param u: Numerical function.
    :param v: Numerical function.
    :return: The vector field Lie bracket [u, v] applied to Id, at x = var, [u, v](Id)(var).
    """
    return u(var) * jax.grad(v)(var) - v(var) * jax.grad(u)(var)


def _vf_gen(bm=get_approx()):
    get_coeffs, eval_fun = bm()

    def vf(drift, sigma, delta, t_k, *coeffs_k):
        dW = coeffs_k[0]
        H = coeffs_k[1] * 1 / jnp.sqrt(6)
        one_lie_bracket = lambda t: lambda z: lie(z, lambda u: sigma(u, t + t_k), lambda u: drift(u, t + t_k))
        second_lie_bracket = lambda t: lambda z: lie(z, lambda u: sigma(u, t + t_k), lambda u: one_lie_bracket(t)(u))
        vector_field = lambda z, t: drift(z, t_k + t) + sigma(z, t_k + t) @ dW / delta + one_lie_bracket(t)(z) @ H \
                                    + second_lie_bracket(t)(z) @ (0.6 @ H @ H.T + 1 / 30 * delta * jnp.eye(dW.shape[0]))

        return vector_field
    return get_coeffs, vf