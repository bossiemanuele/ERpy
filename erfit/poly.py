"""
Polynomial Library Construction
===============================

This module generates polynomial feature libraries for sparse nonlinear system
identification. Given a state trajectory matrix and a maximum polynomial order,
it constructs:

- the feature/design matrix containing monomial evaluations,
- the corresponding exponent (power) matrix defining each monomial term.

The polynomial library is used by ERFit as the candidate set of nonlinear terms
from which sparse governing equations are selected.

Functions
---------
polyspace
    Generate a polynomial feature matrix and exponent matrix up to a given order.

Notes
-----
The exponent matrix returned by this module is especially important for
interpreting identified models and reconstructing human-readable differential
equations.

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

import itertools
import math
import numpy as np


def polyspace(X: np.ndarray, order: int):
    X = np.asarray(X, dtype=float)
    N = X.shape[1]
    if order <= 6:
        return _lowOrder(X, order)
    if ((N <= 5) and (order <= 10)) or (N <= 3):
        exponents = []
        for powers in itertools.product(range(order + 1), repeat=N):
            if sum(powers) <= order:
                exponents.append(tuple(reversed(powers)))
        P = np.unique(np.array(exponents, dtype=int), axis=0)
        A = np.column_stack([np.prod(X ** row, axis=1) for row in P])
        return A, P
    return _lowOrder(X, order)


def _lowOrder(X, order):
    m, n = X.shape
    A = np.ones((m, 1), dtype=float)
    A = np.column_stack([A, X])
    P = np.vstack([np.zeros((1, n), dtype=int), np.eye(n, dtype=int)])
    if order > 1:
        A, P = _add_terms(A, P, X, 2)
    if order > 2:
        A, P = _add_terms(A, P, X, 3)
    if order > 3:
        A, P = _add_terms(A, P, X, 4)
    if order > 4:
        A, P = _add_terms(A, P, X, 5)
    if order > 5:
        A, P = _add_terms(A, P, X, 6)
    return A, P


def _add_terms(A, P, X, degree):
    N = X.shape[1]
    for combo in itertools.combinations_with_replacement(range(N), degree):
        A = np.column_stack([A, np.prod(X[:, combo], axis=1)])
        t = np.zeros(N, dtype=int)
        for idx in combo:
            t[idx] += 1
        P = np.vstack([P, t])
    return A, P
