# Bayesian SDE solvers

Companion code in JAX to the article *** by Yvann Le Fay, Simo Särkkä and Adrien Corenflos.

## What is it?
-----------
This is a [JAX](https://github.com/google/jax) implementation of 1.0 strongly convergent SDE schemes including novel Gaussian-based probabilistic SDE solvers.

## Supported features
-----------
- Classic SDE schemes: Euler-Maruyama, 1.5 Taylor-Itô
- Exotic Gaussian filtering SDE schemes including 1.0 strongly convergent scheme based on piecewise polynomial approximations of the Brownian motion. Can be used both for pathwise and moment computations.
- Euler ODE scheme.
- Extended Kalman filtering, with lower square root implementation.

## Usage
-----------
See the `scripts` and `tests` folders for examples of usage.

## Reproducing the results of the article
-----------
Please refer to `scripts/README.md` for instructions on how to reproduce the results of the article.

## License
-----------
This project is licensed under the MIT License.


