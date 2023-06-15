import numpy as np
from jax import grad
from euler import euler
from foster_polynomial import parabolas
import matplotlib.pyplot as plt

def sde_solver(
        drift,
        sigma,
        x0,
        bm,
        delta,
        N,
        ode_int,
        key,
        batch_size=None,
):
    init = x0
    polys = bm(delta, N, key=key)
    polynomial_vector_fields = np.array([lambda x, t: \
        drift(x, t+delta*k) + sigma(x, t+delta*k) * grad(polys[k])(t) for k in range(N)]) #pb lazy evaluation?
    solution = np.zeros(shape=(N+1,))
    solution[0] = init
    for k in range(1, N+1):
        solution[k] = ode_int(key=key, init=init, vector_field=polynomial_vector_fields[k-1], T=delta)
        init = solution[k]
    return solution

def wrapped_euler(key, init, vector_field, T):
    #10 points euler
    N = 100
    return euler(init=init, vector_field=vector_field, h=T/N, N=N)[-1]

def parabola_sde_solver_euler(drift, sigma, x0, delta, N, key):
    return sde_solver(drift=drift, sigma=sigma, x0=x0, bm=parabolas, delta=delta, N=N, ode_int=wrapped_euler, key=key)

drift = lambda x, t: t
sigma = lambda x, t: 0.0
delta = 0.1
x0 = 1.0
N = 10
key = 1337
sol=parabola_sde_solver_euler(drift, sigma, x0, delta, N, key)
print(sol)
plt.plot(np.linspace(0, delta*N, N+1), sol)
plt.savefig("out.png", dpi=300)