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
    V_L = -70.0
    V_E = 0.0
    V_I = -80.0
    I_inj = -0.6
    V0 = -60.0
    G_E0 = 10.0
    G_I0 = 10.0
    C = G_E0 + G_I0  # total conductance, not important for our tests
    G_L = 50.0

    x0 = jnp.array([V0, G_E0, G_I0])

    N = 100
    h = 1 / N

    def drift(x):
        return jnp.array(
        [
            1.0
            / C
            * (-G_L * (x[0] - V_L) - x[1] * (x[0] - V_E) - x[2] * (x[0] - V_I) + I_inj),
            -1.0 / tau_E * (x[1] - gbar_E),
            -1.0 / tau_I * (x[2] - gbar_I),
        ]
    )
    def sigma(x):
        return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [sig_E * x[1] ** 0.5, 0.0, 0.0],
            [0.0, sig_I * x[2] ** 0.5, 0.0],
        ]
    )

    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    def theoretical_mean_up_to_order_2(k):
        t = k * h
        return (x0 + t * drift(x0) + t ** 2 / 2 * jnp.array(
            [-1.0 / C * (drift(x0)[0] * (G_L + x0[1] + x0[2]) + drift(x0)[1] * (x0[1] - V_E) + drift(x0)[2] * (
                    x0[0] - V_I)),
             -1.0 / tau_E * drift(x0)[1],
             -1.0 / tau_I * drift(x0)[2]])
                )

    linspaces, sols = wrapped_hypoelliptic_15(keys)

    def theoretical_variance_up_to_order_3_first_coordinate(k):
        t = k * h
        return (t ** 3 / (3 * C ** 2) * (
                (x0[0] - V_E) ** 2 * sig_E ** 2 * x0[2]
                + (x0[0] - V_I) ** 2 * sig_I ** 2 * x0[2]))

    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), theoretical_mean_up_to_order_2(1), decimal=2
    )
    npt.assert_almost_equal(
        sols[:, 10, 0].var(),
        theoretical_variance_up_to_order_3_first_coordinate(10),
        decimal=4,
    )


def test_harmonic_oscillator():
    @jax.vmap
    def wrapped_hypoelliptic_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)

    gamma = 1.0
    D = 1.0
    sig = 2.0

    drift = lambda x: jnp.dot(jnp.array([[0.0, 1.0], [-D, -gamma]]), x)
    sigma = lambda x: jnp.array([[0.0, 0.0], [sig, 0.0]])

    x0 = jnp.ones((2,))
    N = 100
    h = 2 / N

    def theoretical_variance_up_to_order3(k):
        t = k * h
        return sig ** 2 * jnp.array(
            [
                [1 / 3 * t ** 3, 1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma],
                [1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma, t - gamma * t ** 2 + 1 / 3 * t ** 3 * (2 * gamma ** 2 - D)]
            ])

    linspaces, sols = wrapped_hypoelliptic_15(keys)
    npt.assert_array_almost_equal(
        jnp.cov(sols[:, 1], rowvar=False),
        theoretical_variance_up_to_order3(1),
        decimal=2,
    )
