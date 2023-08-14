import jax
import jax.numpy as jnp
from functools import partial
from bayesian_sde_solver.foster_polynomial import get_approx

JAX_KEY = jax.random.PRNGKey(1337)

M, N = 4, 1
h = 1.0
K = 1000
bm_key, exp_key = jax.random.split(JAX_KEY, 2)
exp_keys = jax.random.split(exp_key, M * N)
bms = jax.random.normal(bm_key, shape=(M * K - 1, 1, 1))
bms = jnp.insert(bms, 0, 0, axis=0)
bms = bms * h ** 0.5 / K ** 0.5
bms = jnp.cumsum(bms, axis=0)

#given brownian increments, generate parabolas

@jax.jit
def path(bms_inc):

    parabola_approx, eval_approx = get_approx()

    _, b = parabola_approx(exp_keys, h)
    b = b.reshape(M, N, 1)

    linspace01 = jnp.linspace(0, h, K + 1)
    bms_inc2 = jnp.diff(bms_inc, axis=0)
    results = eval_approx(linspace01, h, bms_inc2, b)

    results = bms_inc[:-1] + results

    return results

bms_inc = bms[::K-1]
results = path(bms_inc)
linspace01 = jnp.linspace(0, h, K + 1)
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=jnp.linspace(0, M*h, (K-1)*M+1), y=bms[...,0,0], mode='lines', name='Brownian'))
for k in range(M):
    for n in range(N):
        fig.add_trace(go.Scatter(x=linspace01 + float(h * k), y=results[k, n, :], mode='lines',
                                 opacity=0.1))
fig.show()
