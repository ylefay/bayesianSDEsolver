import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="bayesian_sde_solver",
    author="Yvann Le Fay, Adrien Corenflos",
    description="Bayesian SDE solver",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "chex>=0.1.5",
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "matplotlib>=3.6.3"
        "pytest",
        "statsmodels>=0.13.5",
        "tensorflow_probability>=0.19.0",
        "tqdm>=4.64.1",
        "numpy>=1.24.3",
        "probdiffeq>=0.1.4",
    ],
    long_description_content_type="text/markdown",
    keywords="probabilistic state space bayesian statistics sampling algorithms sde",
    license="MIT",
    license_files=("LICENSE",),
)