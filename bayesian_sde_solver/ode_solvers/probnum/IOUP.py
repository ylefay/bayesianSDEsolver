from math import factorial
from typing import Tuple

import jax.numpy as jnp
import jax.scipy.linalg as linalg
from numpy.typing import ArrayLike


def transition_function(theta: float, sigma: float, q: int, dt: float, dim: int) -> Tuple[ArrayLike, ArrayLike,
                                                                                          ArrayLike]:
    """
    Closed formula for Integrated Ornstein-Uhlenbeck transition function.
    """
    F = jnp.block(
        [[jnp.zeros(q).reshape((q, 1)), jnp.diag(jnp.ones(q))],
         [jnp.append(jnp.zeros(q), jnp.array([-theta])).reshape((1, q + 1))]]
    )

    A = linalg.expm(F * dt)
    A = jnp.kron(jnp.eye(dim), A)

    Q = sigma ** 2 * jnp.array([
        [dt ** (2 * q + 1 - i - j) / ((2 * q + 1 - i - j) * factorial((q - i)) * factorial((q - j)))
         for j in range(q + 1)] for i in range(q + 1)])  # leading order
    Q = jnp.kron(jnp.eye(dim), Q)
    m = jnp.zeros(dim * (q + 1))

    return m, Q, A
