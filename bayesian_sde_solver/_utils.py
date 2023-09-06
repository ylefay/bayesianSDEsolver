from typing import Tuple

import jax.numpy as jnp


def insert(x, pos, y, axis):
    """
    Wrapper for jnp.insert handling tuples.
    Used for inserting (sample[0], mean[0], variance[0]) inside a prob. num solutions.
    """
    if isinstance(x, Tuple):
        return tuple(jnp.insert(x[i], pos, y[i], axis) for i in range(len(x)))
    else:
        return jnp.insert(x, pos, y, axis)
