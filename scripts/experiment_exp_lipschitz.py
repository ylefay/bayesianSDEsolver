from functools import partial

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

from bayesian_sde_solver.foster_polynomial import get_approx as parabola_approx

# Expected Lipschitz constant of the flow:
# \mathbb{E}[e^{L(delta)}] \leq 1 + K_\sigma(\sqrt{\frac{3}{2\pi}}+\sqrt{\frac{2}{\pi}})\sqrt{\delta}
# Remark D.1

seed = jax.random.PRNGKey(1337)
keys = jax.random.split(seed, 100000)


@jax.jit
@partial(jnp.vectorize, signature="()->(d)")
def experiment(delta):
    N = 100
    linspace = jnp.linspace(0, delta, N + 1)
    get_coeffs, eval_fn = parabola_approx()
    coeffs = jax.vmap(get_coeffs, in_axes=(0, None))(keys, delta)

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def integrand_mean(t, a, b):
        func = lambda z: eval_fn(z, delta, a, b)
        return jnp.abs(jax.jacfwd(func)(t))

    ys = integrand_mean(linspace, *coeffs)
    trapz = jax.vmap(jax.vmap(jnp.trapz, in_axes=[1, None]), in_axes=[2, None])(
        ys, linspace
    )
    exp_trapz = jnp.exp(trapz)
    return exp_trapz.mean(axis=1)


def theoretical_bound(delta):
    return 1 + (jnp.sqrt(3 / (2 * 3.1415)) + jnp.sqrt(2 / 3.1415)) * delta ** 0.5


deltas = jnp.logspace(-5, -4, 10)
res = experiment(deltas)
res2 = theoretical_bound(deltas)
plt.semilogy(deltas, res2, label="theoretical bound")
plt.semilogy(deltas, res, label="experiment")
plt.legend()
plt.show()
