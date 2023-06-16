import jax
import jax.numpy as jnp
from jax import linearize





def drift_correction(diffusion, t, x):
    """diff_val, diff_jvp = linearize(lambda z: diffusion(z, t), x)
    out = jax.vmap(diff_jvp, in_axes=[1])(diff_val)
    return jnp.sum(out, axis=[1, 2])"""
    return jnp.einsum('jk,jik->i', diffusion(x, t), jax.jacfwd(lambda z: diffusion(z, t))(x))

def to_stratonovich(drift, diffusion):
    def new_drift(x, t):
        if jnp.ndim(x) == 0:
            return drift(x, t) - 0.5 * diffusion(x, t) * jax.grad(lambda z: diffusion(z, t))(x)

        return drift(x, t) - 0.5 * drift_correction(diffusion, t, x)

    return new_drift, diffusion


def to_ito(drift, diffusion):
    def new_drift(x, t):
        if jnp.ndim(x) == 0:
            return drift(x, t) + 0.5 * diffusion(x, t) * jax.grad(lambda z: diffusion(z, t))(x)
        return drift(x, t) + 0.5 * drift_correction(diffusion, t, x)

    return new_drift, diffusion
