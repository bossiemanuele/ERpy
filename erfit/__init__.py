"""
ERPy: Entropic Regression package in Python
====================

Top-level public API for the ERPy package.

This module exposes the main user-facing functions for entropy-based sparse
system identification, including option handling, entropy and mutual
information estimators, polynomial library generation, regression routines,
ODE reconstruction, and synthetic dataset generation.

Main exports
------------
- eroptset: create and update package options.
- dEntropy, MIDiscrete, MIKnn, tolEstimate: entropy and MI utilities.
- lsfit, iterativeRWLS, erforward, erbackward, erfit: regression and model selection.
- polyspace: polynomial feature library generation.
- getSystemDataset, centralDifference: synthetic data generation and derivative estimation.
- lorenzODE, rosslerODE, vanderpol, getODEHandle: benchmark systems and reconstructed ODE handles.

Notes
-----
This file is intended only to organize and expose the public package interface.
Core algorithmic details are implemented in the corresponding submodules.

References
-----
The methodology implemented in this repository is based on:
AlMomani, A. A. R., Sun, J., & Bollt, E. (2020).
"How entropic regression beats the outliers problem in nonlinear system identification."
Chaos, 30(1), 013107. https://doi.org/10.1063/1.5133386

Author and Affiliation
---------
Emanuele Bossi, Embry-Riddle Aeronautical University

Date Last Modified
---------
04/04/2026

License
-------
MIT

Citation
--------
If you use this package in academic work, please cite:
[Add paper / report / thesis citation here]
"""

from .options import eroptset
from .entropy import dEntropy, discretizeData, MIDiscrete, MIKnn, tolEstimate
from .regression import lsfit, iterativeRWLS, erforward, erbackward, erfit
from .poly import polyspace
from .data_generation import centralDifference, getSystemDataset, lorenzODE, rosslerODE, vanderpol
from .ode import getODEHandle

__all__ = [
    "eroptset", "dEntropy", "discretizeData", "MIDiscrete", "MIKnn", "tolEstimate",
    "lsfit", "iterativeRWLS", "erforward", "erbackward", "erfit",
    "polyspace", "centralDifference", "getSystemDataset",
    "lorenzODE", "rosslerODE", "vanderpol", "getODEHandle"
]