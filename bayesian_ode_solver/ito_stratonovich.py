import jax
import jax.numpy as jnp


def to_stratonovich(drift, diffusion):
    def new_drift(x, t):
        if jnp.ndim(x) == 0:
            return drift(x, t) - 0.5 * diffusion(x, t) * jax.grad(diffusion)(x, t)
        raise NotImplementedError

    return new_drift, diffusion


def to_ito(drift, diffusion):
    def new_drift(x, t):
        if jnp.ndim(x) == 0:
            return drift(x, t) + 0.5 * diffusion(x, t) * jax.grad(diffusion)(x, t)
        raise NotImplementedError

    return new_drift, diffusion
