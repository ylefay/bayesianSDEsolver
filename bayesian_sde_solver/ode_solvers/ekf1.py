import jax
import jax.numpy as jnp
from probdiffeq import ivpsolvers, ivpsolve
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters


def solver(key, init, vector_field, h, N):
    dim = init.shape[0]

    def vf(x, t=0.0, p=None):
        return vector_field(x, t)

    ts0 = ivpsolvers.solver_calibrationfree(
        *filters.filter(
            *recipes.ts1_dense(ode_order=1, num_derivatives=1, ode_shape=(dim,))
        )
    )
    ts = jnp.linspace(0, N * h, N + 1)
    solution = ivpsolve.solve_fixed_grid(
        vector_field=vf, initial_values=(init,), solver=ts0, grid=ts
    )
    if key is None:
        y, samples = (
            solution.marginals.marginal_nth_derivative(0).mean[-1],
            solution.marginals,
        )
    else:
        m = solution.marginals.marginal_nth_derivative(0).mean[-1]
        cholesky = solution.marginals.marginal_nth_derivative(0).cov_sqrtm_lower[-1]
        normal = jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))
        y, samples = m + cholesky @ normal, solution.marginals
    return y  # return only the mean of the last point of the trajectory, you may want the covariance as well
