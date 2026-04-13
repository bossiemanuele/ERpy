"""
ODE Reconstruction Utilities
============================

This module provides utilities for converting an identified sparse coefficient
matrix and polynomial exponent matrix into a callable ordinary differential
equation (ODE) right-hand side.

The main function, `getODEHandle`, constructs a Python function compatible with
standard ODE solvers. The returned handle evaluates the recovered dynamical
system at any state vector by assembling the active polynomial terms and their
identified coefficients.

Functions
---------
getODEHandle
    Build a callable ODE function from a polynomial library and sparse
    coefficient matrix.

Notes
-----
This module is useful for validating identified models through simulation,
trajectory comparison, and qualitative phase-space analysis.

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

import numpy as np


def getODEHandle(P, S):
    P = np.asarray(P, dtype=int)
    S = np.asarray(S, dtype=float)
    def fhandle(t, x):
        x = np.asarray(x, dtype=float)
        vals = []
        for i in range(S.shape[1]):
            ix = S[:, i] != 0
            p = P[ix, :]
            B = S[ix, i]
            if p.size == 0:
                vals.append(0.0)
                continue
            Phi = np.array([np.prod(x ** row) for row in p], dtype=float)
            vals.append(float(np.dot(B, Phi)))
        return np.array(vals)
    return fhandle
