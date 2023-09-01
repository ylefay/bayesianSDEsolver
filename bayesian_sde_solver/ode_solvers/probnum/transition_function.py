from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
from numpy.typing import ArrayLike


def transition_function(F: jnp.array, u: jnp.array, L: jnp.array, h: float, n_linspace=10000) -> Tuple[
    ArrayLike, ArrayLike,
    ArrayLike]:
    r"""
    A prior of the form
        \mathrm{d}X(t) = (FX(t) + u)\mathrm{d}t + L \mathrm{d}W_t,
    has the following strong solution:
        X(t+h) = \exp{Fh}(X(t) + \int_0^h \exp{-Fs}L \mathrm{d}W_s),
    where
        X(t+h) \mid X(t) ~ \mathcal{N}(A(h)X(t) + xi(h), Q(h)).
    ----------------------------
    Return \xi(h), Q(h), A(h).
    """
    linspace = jnp.linspace(0, h, n_linspace)
    A = linalg.expm(F * h)

    @jax.vmap
    def integrand_xi(s):
        return linalg.expm(F * s) @ u

    integrand_xi_values = integrand_xi(linspace)
    xi = jnp.trapz(integrand_xi_values, linspace, axis=0)

    @jax.vmap
    def integrand_Q(s):
        B = linalg.expm(F * s) @ L
        return B @ B.T

    integrand_Q_values = integrand_Q(linspace)
    Q = jnp.trapz(integrand_Q_values, linspace, axis=0)

    return xi, Q, A
