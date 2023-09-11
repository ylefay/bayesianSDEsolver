import jax
import jax.numpy as jnp


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

    return x0, drift, sigma


def square_matrix_fhn():
    """
    Make the matrix diffusion squared to use ssm_parabola_ode.
    """
    x0, drift, _sigma = fhn()

    def sigma(x, t):
        return jnp.array([[0.0, 0.0], [_sigma(x, t)[1, 0], 0.0]])

    return x0, drift, sigma


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

    return x0, drift, sigma


def square_matrix_ibm():
    x0, drift, _sigma = ibm()

    def sigma(x, t):
        return jnp.array([[0.0, 0.0], [_sigma(x, t)[1, 0], 0.0]])

    return x0, drift, sigma


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
