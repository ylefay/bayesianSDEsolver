from math import factorial
from typing import Callable, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as linalg


def transition_function(theta: float, sigma: float, q: int, dt: float, dim: int) -> Tuple[Callable, jnp.ndarray, jnp.ndarray]:
    F = jnp.block(
        [[jnp.zeros(q).reshape((q, 1)), jnp.diag(jnp.ones(q))],
         [jnp.append(jnp.zeros(q), jnp.array([-theta])).reshape((1, q + 1))]]
    )
    #F = jnp.kron(jnp.eye(dim), F)

    A = linalg.expm(F * dt)
    A = jnp.kron(jnp.eye(dim), A)

    Q = sigma ** 2 * jnp.array([
            [dt ** (2 * q + 1 - i - j) / ((2 * q + 1 - i - j) * factorial((q - i)) * factorial((q - j)))
                for j in range(q + 1)] for i in range(q + 1)])
    Q = jnp.kron(jnp.eye(dim), Q)
    m = jnp.zeros(dim * (q + 1))

    def transition(x):
        return jnp.dot(A, x)

    return transition, m, Q
