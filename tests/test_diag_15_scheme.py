import jax
import jax.numpy as jnp
import numpy.testing as npt

from bayesian_sde_solver.sde_solvers import diag_15_scheme
from bayesian_sde_solver.utils.ivp import synaptic_conductance_reduced


def test_synaptic_conductance():
    JAX_KEY = jax.random.PRNGKey(1337)
    keys = jax.random.split(JAX_KEY, 1_000)
    N = 100
    h = 1 / N
    x0, drift, sigma, theoretical_mean_up_to_order_2 = synaptic_conductance_reduced()

    @jax.vmap
    def wrapped_15(key_op):
        return diag_15_scheme(key=key_op, init=x0, drift=drift, sigma=sigma, h=h, N=N)

    linspaces, sols = wrapped_15(keys)

    npt.assert_array_almost_equal(
        jnp.mean(sols[:, 1], axis=0), theoretical_mean_up_to_order_2(h), decimal=2
    )
