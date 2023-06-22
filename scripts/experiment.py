import jax
import jax.numpy as jnp

from bayesian_sde_solver.foster_polynomial import get_approx

JAX_KEY = jax.random.PRNGKey(1337)
M, N = 2, 100
DT = 1
K = 500
bm_key, exp_key = jax.random.split(JAX_KEY, 2)
exp_keys = jax.random.split(exp_key, M * N).reshape(M, N, 2)

bms = jax.random.normal(bm_key, shape=(M, K - 1))
bms = jnp.insert(bms, 0, 0, axis=1)
bms = bms / (K * M * DT) ** 0.5
bms_cums = jnp.cumsum(bms, axis=1)
bms_inc = bms_cums[:, -1]
bms_incrbis = jnp.diff(bms_inc[:: K])

parabola_approx, eval_approx = get_approx()
_, b = parabola_approx(exp_keys, 1 / K)
linspace = jnp.linspace(0, M, M * K).reshape(-1, K, 1)
linspace01 = jnp.linspace(0, 1, K + 1)
results = eval_approx(linspace01, DT, bms_cums[:, K - 1::], b)
print(results)
