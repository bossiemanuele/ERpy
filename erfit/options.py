"""
Package Options and Default Configuration
=========================================

This module defines the default option structure used throughout the ERFit
package. It provides a centralized interface for setting numerical,
information-theoretic, and regression-related hyperparameters.

The main entry point is `eroptset`, which returns a dictionary of default
options and allows user-provided keyword arguments to override selected values.

Options controlled here include:
- regression estimator,
- mutual-information estimator,
- binning configuration,
- nearest-neighbor settings,
- embedded shuffle testing parameters,
- user-specified terms to keep during model selection.

Functions
---------
eroptset
    Create and customize the ERFit options dictionary.

Notes
-----
Centralizing configuration in this module improves reproducibility,
maintainability, and consistency across experiments.

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
If you use this module in academic work, please cite:
[Add paper / report / thesis citation here]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Any
import math


def eroptset(**kwargs) -> dict[str, Any]:
    from .regression import lsfit
    from .entropy import MIKnn

    options = {
        "useparallel": False,
        "h": 0.01,
        "keepin": [],
        "grayModelEstimator": lsfit,
        "skipForward": False,
        "fkeepin": [],
        "MIEstimator": MIKnn,
        "BinMethod": "fixed",
        "numBins": 16,
        "p": math.inf,
        "K": 2,
        "EmbeddedShuffleTest": False,
        "alpha": 0.99,
        "numPerm": 200,
    }
    options.update(kwargs)
    return options
