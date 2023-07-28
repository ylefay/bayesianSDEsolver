import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import diag_15_scheme


def test_synaptic_conductance():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    tau_E = 0.5
    tau_I = 1.0
    gbar_E = 17.8
    gbar_I = 9.4
    sig_I = 0.1
    sig_E = 0.1
    G_E0 = 10.0
    G_I0 = 10.0

    x0 = jnp.array([G_E0, G_I0])

    N = 100
    h = 1 / N

    def drift(x):
        return jnp.array(
            [-1.0 / tau_E * (x[1] - gbar_E), -1.0 / tau_I * (x[2] - gbar_I)]
        )

    def sigma(x):
        return jnp.array(
            [[sig_E * x[1] ** 0.5, 0.0], [0.0, sig_I * x[2] ** 0.5]]
        )

    @jax.vmap
    def wrapped_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    def theoretical_mean_up_to_order_2(k):
        t = k * h
        return x0 + t * drift(x0) + t ** 2 / 2 * jnp.array([-1.0 / tau_E * drift(x0)[1], -1.0 / tau_I * drift(x0)[2]])

    linspaces, sols = wrapped_15(keys)

    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), theoretical_mean_up_to_order_2(1), decimal=2
    )