# ERpy

ERpy is a lightweight Python implementation of **Entropic Regression (ER)** for sparse identification of nonlinear dynamical systems.  
The repository reproduces the core ER workflow described in AlMomani, Sun, and Bollt (2020), combining regression with **mutual information (MI)** and **conditional mutual information (CMI)** to select informative terms from a candidate function library.

In ER, model selection is performed through a forward-backward procedure:
- **forward selection** adds candidate terms that provide the largest information gain,
- **backward elimination** removes terms that become redundant once other terms are included.

This repository provides:
- the `erfit` Python package,
- scripts to reproduce the main numerical results,
- a simple end-to-end demonstration script,
- a basic automated test.

---

## Installation

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

The package metadata is defined in `pyproject.toml`, and the package is installed under the module name:

```python
import erfit
```

Main dependencies:
- `numpy`
- `scipy`

Some figure-generation scripts also use:
- `matplotlib`
- `scikit-learn`

---

## Quick Start

A minimal example using the Lorenz system is:

```python
import numpy as np
from erfit import getSystemDataset, erfit, eroptset, lorenzODE

np.random.seed(1)

# Generate synthetic data
data = getSystemDataset(lorenzODE)

# Set ER options
opts = eroptset()

# Run entropic regression
S, info = erfit(data["Phi"], data["Xdot"], opts)

print("Recovered coefficient matrix:")
print(S)
```

Depending on your workflow, you may also use:
- `polyspace` to build polynomial libraries,
- `centralDifference` to estimate derivatives from measured trajectories,
- `getODEHandle` to reconstruct identified dynamics,
- `MIDiscrete` or `MIKnn` for mutual-information estimation.

---

## Reproducing the Main Scripts

This repository includes three top-level scripts.

### `Test.py`
End-to-end demonstration and validation script for the ER workflow.  
It generates synthetic benchmark data, runs ER, and prints recovered equations.

Run with:

```bash
python Test.py
```

### `Figure1.py`
Reproduces the Lorenz-system benchmark comparing:
- LS
- LASSO
- SINDy
- ER

This script is intended to recreate the main comparison figure discussed in the paper.

Run with:

```bash
python Figure1.py
```

### `Figure2.py`
Compares two mutual-information estimators within ER:
- binning-based MI
- k-nearest-neighbor MI

This script generates the estimator-comparison figure discussed in the paper.

Run with:

```bash
python Figure2.py
```

### `tests/test_basic.py`
Basic automated test file for the package.

If you use `pytest`, run:

```bash
pytest tests/test_basic.py
```

---

## Repository Structure

```text
ERpy/
├── article/
│   ├── fig/
│   ├── ERpy_Paper.pdf
│   ├── biblio.bib
│   ├── content.tex
│   └── metadata.yaml
├── erfit/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── entropy.py
│   ├── ode.py
│   ├── options.py
│   ├── poly.py
│   └── regression.py
├── paper_reproduction/
│   ├── Figure1.py
│   ├── Figure2.py
│   ├── Test.py
│   └── test.md
├── tests/
│   └── test_basic.py
├── .gitattributes
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Package Contents

The `erfit` package exposes the main ER functionality through:

- `eroptset`  
  Create and modify package options.

- `dEntropy`, `discretizeData`, `MIDiscrete`, `MIKnn`, `tolEstimate`  
  Entropy and mutual-information utilities.

- `lsfit`, `iterativeRWLS`, `erforward`, `erbackward`, `erfit`  
  Regression and ER model-selection routines.

- `polyspace`  
  Polynomial library generation.

- `centralDifference`, `getSystemDataset`  
  Synthetic data generation and derivative estimation.

- `lorenzODE`, `rosslerODE`, `vanderpol`, `getODEHandle`  
  Benchmark dynamical systems and reconstructed ODE handles.

---

## Important Note on Reproducibility

This implementation is intended to closely follow the original MATLAB ERFit workflow presented in AlMomani et al. (2020).  
Even when the methodology is matched carefully, small numerical differences may still arise due to differences between Python and MATLAB in:
- ODE solvers,
- pseudoinverse and linear algebra backends,
- random number generation,
- histogram/binning details,
- regression internals.

For that reason, reproducibility should be evaluated primarily in terms of **consistent qualitative trends and comparable quantitative behavior**, rather than exact coefficient-by-coefficient equality across platforms.

---

## Reference

If you use this repository, please cite the accompanying paper:

```text
Bossi, E. (2026). ERpy: Python Implementation of Entropic Regression for Nonlinear System Identification.
```

Primary methodological reference:

```text
AlMomani, A. A. R., Sun, J., & Bollt, E. (2020).
How entropic regression beats the outliers problem in nonlinear system identification.
Chaos, 30(1), 013107.
```

---

## License

This project is released under the MIT License. See `LICENSE.txt` for details.
