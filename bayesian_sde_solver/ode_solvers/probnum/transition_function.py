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
        X(t+h) \mid X(t) ~ \mathcal{N}(A(h)X(t) + \xi(h), Q(h)).
    ----------------------------
    Return \xi(h), Q(h), A(h).
    """
    linspace = jnp.linspace(0, h, n_linspace)
    A = linalg.expm(F * h)

    @jax.vmap
    def integrand_xi(s):
        return linalg.expm(F * s) @ u

    integrand_xi_values = integrand_xi(linspace)
    xi = jnp.trapezoid(integrand_xi_values, linspace, axis=0)

    @jax.vmap
    def integrand_Q(s):
        B = linalg.expm(F * s) @ L
        return B @ B.T

    integrand_Q_values = integrand_Q(linspace)
    Q = jnp.trapezoid(integrand_Q_values, linspace, axis=0)

    return xi, Q, A


def pad_prior(transition: Tuple[ArrayLike, ArrayLike, ArrayLike], q: int):
    """
    Given a transition, i.e., a tuple (xi, Q, A) and an order q,
    Return the same transition but with 0 padded to make it of order q, i.e., A and Q of size (q+1) \times (q+1) and xi of size q+1.
    """
    xi, Q, A = transition
    q = q + 1
    assert q > xi.shape[0]
    xi = jnp.concatenate([xi, jnp.zeros(q - xi.shape[0])])
    Q = jnp.pad(Q, ((0, q - Q.shape[0]), (0, q - Q.shape[1])))
    A = jnp.pad(A, ((0, q - A.shape[0]), (0, q - A.shape[1])))
    return xi, Q, A


def get_independently_factorized_prior(transitions: Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike]]):
    """
    Given a sequence of (xi, Q, A) tuples, each(xi, Q, A) corresponding to a transition,
    Return the corresponding transition for the concatened state space when each state is,
    a priori, assumed to be independent of the others.
    """
    xi = jnp.concatenate([t[0] for t in transitions])
    Q = jax.scipy.linalg.block_diag(*[t[1] for t in transitions])
    A = jax.scipy.linalg.block_diag(*[t[2] for t in transitions])
    return xi, Q, A
