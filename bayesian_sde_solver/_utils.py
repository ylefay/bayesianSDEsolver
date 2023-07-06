import jax.numpy as jnp
from parsmooth import MVNSqrt, MVNStandard


def insert(x, pos, y, axis):
    """
    Wrapper for jnp.insert handling MVNSqrt and MVNStandard objects.
    """
    if isinstance(x, (MVNStandard, MVNSqrt)):
        return type(x)(jnp.insert(x[0], pos, y[0], axis),
                       jnp.insert(x[1], pos, y[1], axis))
    else:
        return jnp.insert(x, pos, y, axis)
