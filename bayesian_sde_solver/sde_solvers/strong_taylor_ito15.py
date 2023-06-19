import jax
import jax.numpy as jnp
import numpy as np


def solver(key, init, drift, sigma, h, N): #unidimensional case, d=m=1
    dim_noise = sigma(init, 0.).shape[1]
    dim_x = init.shape[0]
    assert dim_x == 1 and dim_noise == 1

    def body(x, inp):
        key_k, t = inp
        bm_key = jax.random.split(key_k, 1)
        dZ, dW = jax.random.multivariate_normal(bm_key, jnp.zeros(2, ),
                                                jnp.array([[h, 0.5 * h ** 2], [0.5 * h ** 2, h ** 3 / 3]]))

        drift_value = drift(x, t)
        sigma_value = sigma(x, t)
        drift_jac = jax.jacfwd(lambda x: drift(x, t))
        sigma_jac = jax.jacfwd(lambda x: sigma(x, t))

        out = x + h * drift_value + sigma_value @ dW + 0.5 * sigma_value @ sigma_jac(x) * \
              (dW ** 2 - h) + drift_jac(x) @ sigma_value * dZ + 0.5 * (drift_value @ drift_jac(x)) + \
              0.5 * sigma_value @ sigma_value @ jax.jacfwd(drift_jac)(x) * h ** 2 + \
              (drift_value @ sigma_jac(x) + 0.5 * drift_value @ drift_value @ jax.jacfwd(sigma_jac)(x)) * (
                          dW * h - dZ) + \
              0.5 * sigma_value @ (sigma_value @ jax.jacfwd(sigma_jac)(x) + sigma_jac(x) @ sigma_jac(x)) * (
                          1 / 3 * dW ** 2 - h) * dW

        return out, out

    keys = jax.random.split(key, N)
    ts = np.linspace(0, N * h - h, N)
    inps = keys, ts
    _, samples = jax.lax.scan(body, init, inps)
    samples = jnp.insert(samples, 0, init, axis=0)
    return ts, samples

