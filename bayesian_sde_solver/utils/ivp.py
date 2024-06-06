import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

"""
Different initial value problems including
    - FitzHugh-Nagumo
    - Multidim. linear SDE
    - Synaptic conductance
    - Harmonic oscillator
    - Geometric Brownian motion
    - Integrated Brownian motion
with for some, local mean and var. of the density transition
or closed-form expressions of the mean and variance.

For 1.5 Taylor-Ito scheme, discard the time argument and use squared-matrix diffusion.
Use squared-matrix diffusion for SSM scheme.
"""


def fhn():
    """
    FitzHugh-Nagumo IVP.
    """
    gamma = 1.5
    sig = 0.3
    eps = 0.1
    alpha = 0.8
    s = 0.0
    x0 = jnp.zeros((2,))

    def drift(x, t):
        return (jnp.array([[1.0 / eps, -1.0 / eps], [gamma, -1]]) @ x + jnp.array(
            [s / eps - x[0] ** 3 / eps, alpha]))

    def sigma(x, t):
        return jnp.array([[0.0], [sig]])

    def theoretical_mean_up_to_order_2(t):
        return x0 + t * jnp.array(
            [1 / eps * (x0[0] - x0[0] ** 3 - x0[1] + s) + t / 2 * 1 / eps * (
                    1 / eps * (1 - 3 * x0[0] ** 2) * (x0[0] - x0[0] ** 3 - x0[1] + s)
                    - (gamma * x0[0] - x0[1] - alpha)),
             gamma * x0[0] - x0[1] + alpha + t / 2 * (
                     gamma / eps * (x0[0] - x0[0] ** 3 - x0[1] + s) - (gamma * x0[0] - x0[1] + alpha))])

    def theoretical_variance_up_to_order3(t):
        return sig ** 2 * jnp.array(
            [
                [1 / 3 * t ** 3 * 1 / eps ** 2, -1 / 2 * t ** 2 * 1 / eps],
                [-1 / 2 * t ** 2 * 1 / eps, t - t ** 2],
            ])

    return x0, drift, sigma, theoretical_mean_up_to_order_2, theoretical_variance_up_to_order3


def square_matrix_fhn():
    """
    Make the matrix diffusion squared to use ssm_parabola_ode or 1.5 scheme (in that case, discard
    the second argument).
    Theoretical mean and variance of local transition density
    """
    x0, drift, _sigma, theoretical_mean_up_to_order_2, theoretical_variance_up_to_order3 = fhn()

    def sigma(x, t):
        return jnp.array([[0.0, 0.0], [_sigma(x, t)[1, 0], 0.0]])

    return x0, drift, sigma, theoretical_mean_up_to_order_2, theoretical_variance_up_to_order3


def ibm():
    """
    Integrated Brownian motion IVP.
    """
    sig = 1.0
    x0 = jnp.ones((2,))

    def drift(x, t):
        return jnp.array([[0., 1.0], [0., 0.]]) @ x

    def sigma(x, t):
        return jnp.array([[0.0], [sig]])

    def var(t):
        return jnp.array([[t ** 3 / 3, t ** 2 / 2],
                          [t ** 2 / 2, t]]) * sig ** 2

    return x0, drift, sigma, var


def double_ibm():
    """
    Double integrated Brownian motion IVP.
    """
    sig = 1.0
    x0 = jnp.ones((3,))

    def drift(x, t):
        return jnp.array([[0., 1.0, 0.], [0., 0., 1.0], [0., 0., 0.]]) @ x

    def sigma(x, t):
        return jnp.array([[0.0], [0.0], [sig]])

    def var(t):
        return jnp.array([[t ** 5 / 20, t ** 4 / 8, t ** 3 / 6],
                          [t ** 4 / 8, t ** 3 / 3, t ** 2 / 2],
                          [t ** 3 / 6, t ** 2 / 2, t]]) * sig ** 2

    return x0, drift, sigma, var


def square_matrix_ibm():
    x0, drift, _sigma, var = ibm()

    def sigma(x, t):
        return jnp.array([[0.0, 0.0], [_sigma(x, t)[1, 0], 0.0]])

    return x0, drift, sigma, var


def random_linear_sde(key=jax.random.PRNGKey(1337), dim=1, std1=None, std2=None):
    """
    Random linear SDE of the form:
    dX(t) = MX(t)dt + sigma dW(t)
    where M is a dim-squared random matrix generated using random normal variables.
    """
    if std1 is None:
        std1 = jnp.identity(dim)
    if std2 is None:
        std2 = jnp.identity(dim)

    x0 = jnp.ones((dim,))

    drift_matrix = jax.random.normal(key, shape=(dim, dim))

    sigma_matrix = jax.random.normal(key, shape=(dim, dim))

    def drift(x, t):
        return std1 @ drift_matrix @ x

    def sigma(x, t):
        return std2 @ sigma_matrix

    return x0, drift, sigma


def synaptic_conductance_reduced():
    """
    Reduced synaptic conductance IVP,
    And local expansion at order 2 of the mean
    """

    tau_E = 0.5
    tau_I = 1.0
    gbar_E = 17.8
    gbar_I = 9.4
    sig_I = 0.1
    sig_E = 0.1
    G_E0 = 10.0
    G_I0 = 10.0

    x0 = jnp.array([G_E0, G_I0])

    def drift(x):
        return jnp.array(
            [-1.0 / tau_E * (x[1] - gbar_E), -1.0 / tau_I * (x[2] - gbar_I)]
        )

    def sigma(x):
        return jnp.array(
            [[sig_E * x[1] ** 0.5, 0.0], [0.0, sig_I * x[2] ** 0.5]]
        )

    def theoretical_mean_up_to_order_2(t):
        return x0 + t * drift(x0) + t ** 2 / 2 * jnp.array([-1.0 / tau_E * drift(x0)[1], -1.0 / tau_I * drift(x0)[2]])

    return x0, drift, sigma, theoretical_mean_up_to_order_2


def harmonic_oscillator_square():
    """
    Harmonic oscillator IVP
    with theoretical mean and variance.
    Square diffusion matrix for SSM parabola scheme or 1.5 exact diffusion scheme.
    """

    gamma = 1.0
    D = 1.0
    sig = 2.0

    C = jnp.array([[0.0, 0.0], [sig, 0.0]])
    M = jnp.array([[0.0, 1.0], [-D, -gamma]])

    drift = lambda x: jnp.dot(M, x)
    sigma = lambda x: C

    x0 = jnp.ones((2,))

    def theoretical_variance_up_to_order3(t):
        return sig ** 2 * jnp.array(
            [
                [1 / 3 * t ** 3, 1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma],
                [1 / 2 * t ** 2 - 1 / 2 * t ** 3 * gamma, t - gamma * t ** 2 + 1 / 3 * t ** 3 * (2 * gamma ** 2 - D)]
            ])

    def theoretical_mean(t):
        mean = jlinalg.expm(M * t) @ x0
        return mean

    def theoretical_variance(t):
        @jax.vmap
        def integrand_var(s):
            B = jlinalg.expm(M * s) @ C
            return B @ B.T

        linspace_int = jnp.linspace(0, t, 1000)
        var = jnp.trapz(integrand_var(linspace_int), linspace_int, axis=0)
        return var

    return x0, drift, sigma, theoretical_mean, theoretical_variance, theoretical_variance_up_to_order3


def harmonic_oscillator():
    """
    Harmonic oscillator IVP
    And theoretical exact mean,
    and approximately integrated variance.
    This is a linear SDE.
    """
    gamma = 0.1
    D = 0.1
    sig = 1.0

    M = jnp.array([[0.0, 1.0], [-D, -gamma]])
    C = jnp.array([[0.0], [sig]])

    def drift(x, t):
        return jnp.dot(M, x)

    def sigma(x, t):
        return C

    x0 = jnp.ones((2,))

    def theoretical_mean(t):
        mean = jlinalg.expm(M * t) @ x0
        return mean

    def theoretical_variance(t):
        @jax.vmap
        def integrand_var(s):
            B = jlinalg.expm(M * s) @ C
            return B @ B.T

        linspace_int = jnp.linspace(0, t, 1000)
        var = jnp.trapz(integrand_var(linspace_int), linspace_int, axis=0)
        return var

    return x0, drift, sigma, theoretical_mean, theoretical_variance


def gbm():
    """
    Geometric Brownian motion IVP
    With theoretical mean and variance.
    """
    a = 1
    b = 1

    def drift(x, t):
        return a * x

    def sigma(x, t):
        return b * jnp.diag(x)

    x0 = jnp.ones((1,))

    def theoretical_mean(t):
        return x0 * jnp.exp(a * t)

    def theoretical_variance(t):
        return x0 ** 2 * (jnp.exp(b ** 2 * t) - 1) * jnp.exp(2 * a * t)

    return x0, drift, sigma, theoretical_mean, theoretical_variance


def synaptic_conductance():
    """
    Synaptic conductance IVP
    With transition density theoretical mean and variance
    """
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

    def theoretical_mean_up_to_order_2(t):
        return (x0 + t * drift(x0) + t ** 2 / 2 * jnp.array(
            [-1.0 / C * (drift(x0)[0] * (G_L + x0[1] + x0[2]) + drift(x0)[1] * (x0[1] - V_E) + drift(x0)[2] * (
                    x0[0] - V_I)),
             -1.0 / tau_E * drift(x0)[1],
             -1.0 / tau_I * drift(x0)[2]])
                )

    def theoretical_variance_up_to_order_3_first_coordinate(t):
        return (t ** 3 / (3 * C ** 2) * (
                (x0[0] - V_E) ** 2 * sig_E ** 2 * x0[2]
                + (x0[0] - V_I) ** 2 * sig_I ** 2 * x0[2]))

    return x0, drift, sigma, theoretical_mean_up_to_order_2, theoretical_variance_up_to_order_3_first_coordinate
