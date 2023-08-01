import jax

from bayesian_sde_solver.foster_polynomial import get_approx


def _vf_gen(bm=get_approx()):
    get_coeffs, eval_fn = bm()

    def vf(drift, sigma, delta, t_k, *coeffs_k):
        func = lambda t: eval_fn(t, delta, *coeffs_k)
        vector_field = lambda z, t: drift(z, t_k + t) + sigma(z, t_k + t) @ jax.jacfwd(func)(t)
        return vector_field

    return get_coeffs, vf
