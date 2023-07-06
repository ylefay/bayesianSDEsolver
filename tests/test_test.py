from bayesian_sde_solver.ito_stratonovich import to_stratonovich
import jax.numpy as jnp
import jax


x0 = jnp.ones((1,))
vector_field = lambda x, t: x

Cdot = jnp.array([0., 1.]).reshape((1, 2))
C = jnp.array([1., 0.]).reshape((1, 2))


def measure(x, t):
    return Cdot @ x - vector_field(C@x, t)

init = jnp.array([x0, vector_field(x0, 0.)]).reshape((2, 1))
init_var = jnp.zeros((2, 2))

h = 0.5

from bayesian_sde_solver.ode_solvers.probnum import IOUP_transition_function
transition, xi, Q, A = IOUP_transition_function(0.0, 1.0, 1, h, 1)

Htilde = Cdot

m1m = transition(init)
P1m = A @ init_var @ A.T + Q
z1 = measure(m1m, h)
S1 = Htilde @ P1m @ Htilde.T
K1 = P1m @ Htilde.T @ jnp.linalg.inv(S1)
m1 = m1m + K1 @ z1
P1 = P1m - P1m @ K1 @ Htilde

pass