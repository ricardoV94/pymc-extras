# Welcome to `pymc-extras`
<a href="https://gitpod.io/#https://github.com/pymc-devs/pymc-extras">
  <img
    src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod"
    alt="Contribute with Gitpod"
  />
</a>
<img
  src="https://codecov.io/gh/pymc-devs/pymc-extras/branch/main/graph/badge.svg"
  alt="Codecov Badge"
/>

PyMC Extras extends [PyMC](https://www.pymc.io) with additional distributions, inference methods, and model transformations.
It is maintained by the PyMC team and hosts functionality that is too specialized for the core library, but useful enough that you shouldn't have to write it yourself.

Highlights include:

- Automatic marginalization: exact for finite discrete and conjugate variables, approximate via the Laplace approximation
- Alternative inference methods: Pathfinder, DADVI, INLA, Laplace approximation, and better MAP estimation
- Statespace models: SARIMAX, VARMAX, ETS, and structural time series with Kalman filtering
- Additional distributions such as `DiscreteMarkovChain`, `GeneralizedPoisson`, and `GenExtreme`

`pymc-extras` mirrors the namespaces in `pymc` to make usage and migration as easy as possible.
For example, distributions are used exactly like those in `pymc`:

```python
import pymc as pm
import pymc_extras as pmx

with pm.Model():
    xi = pm.HalfNormal("xi", 0.2)
    pmx.GenExtreme("llik", mu=1, sigma=0.5, xi=xi, observed=data)
```

See the [documentation](https://pymc-extras.readthedocs.io/) for the full API reference.

## Installation

```bash
pip install pymc-extras
```

or for the development version:

```bash
pip install git+https://github.com/pymc-devs/pymc-extras.git
```

## Questions

### What belongs in `pymc-extras`?

- statistical methods, for example step methods or model construction helpers
- distributions that are tricky to sample from or test
- specialized fitting methods or distributions
- any code that requires additional optimization before it can be used in practice

Functionality that proves widely useful may graduate to the main `pymc` repository.

### What does not belong in `pymc-extras`?
- Case studies
- Implementations that cannot be applied generically, for example because they are tied to variables from a toy example

## Contributing

We welcome contributions! Check out the [contributing guidelines](https://github.com/pymc-devs/pymc-extras/blob/main/CONTRIBUTING.md) to get started.
