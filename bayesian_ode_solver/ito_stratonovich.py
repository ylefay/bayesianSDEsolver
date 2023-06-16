import jax
import jax.numpy as jnp





def drift_correction(diffusion, t, x):
    return jnp.einsum('jk,ikj->i', diffusion(x, t), jax.jacfwd(lambda z: diffusion(z, t))(x))

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
