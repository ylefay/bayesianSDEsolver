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
    x0, drift, _sigma = fhn()

    def sigma(x, t):
        return jnp.array([[0.0, 0.0], [_sigma(x, t)[1,0], 0.0]])

    return x0, drift, sigma
