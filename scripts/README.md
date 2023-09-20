## pathwise_convergence_experiment_2_RAM.py
Gaussian SDE Filter (Algorithm 2) implementation.
This script is used to do pathwise comparisons of Algorithm. 2 and Euler-Maruyama scheme.
This script is similar to pathwise_convergence_experiment_2.py but uses less RAM by computing the Euler-Maruyama solution on the fly instead of storing Brownian increments in memory and then computing the Euler-Maruyama solution.
See Section 3.1

## pathwise_convergence_experiment_2_RAM_2.py
Gaussian Mixture SDE Filter (Algorithm 3) implementation.
Similar to the previous scheme.
Propagating uncertainty through the posterior variance. 
Very similar convergence rates to Algorithm 2.
See Section 3.2

## pathwise_convergence_experiment_3.py
Marginalised Gaussian SDE Filter (Algorithm 4) implementation.
This is a low-order weakly convergent scheme.
However, it allows to compute exact (under our model) transition densities.
See Section 3.3

## Untitled.ipynb
Reading .npy output files containing paths, to compute both strong and weak errors.

