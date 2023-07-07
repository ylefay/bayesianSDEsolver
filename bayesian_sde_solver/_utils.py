from typing import Tuple

import jax.numpy as jnp


def insert(x, pos, y, axis):
    """
    Wrapper for jnp.insert handling tuples.
    """
    if isinstance(x, Tuple):
        return (jnp.insert(x[0], pos, y[0], axis),
                jnp.insert(x[1], pos, y[1], axis))
    else:
        return jnp.insert(x, pos, y, axis)
