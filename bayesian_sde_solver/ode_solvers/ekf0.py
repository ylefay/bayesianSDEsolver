import jax.numpy as jnp
from probdiffeq import ivpsolvers, ivpsolve
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters

def solver(init, vector_field, h, N):
    def vf(x, t=0.0, p=None):
        return vector_field(x, t)

    ts0 = ivpsolvers.solver_calibrationfree(*filters.filter(*recipes.ts0_iso(ode_order=1, num_derivatives=1)))
    ts = jnp.linspace(0, N * h, N + 1)
    solution = ivpsolve.solve_and_save_at(vector_field=vf, initial_values=(init,), solver=ts0, save_at=ts, dt0=h)
    return solution.marginals.mean[-1, 0, :] # return only the mean of the last point of the trajectory, you may want the covariance as well

