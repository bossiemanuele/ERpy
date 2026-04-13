"""
Basic Unit Tests for the ERFit Package
======================================

This module contains lightweight unit tests covering core functionality of the
ERFit package, including:

- polynomial library construction,
- entropy and mutual-information utilities,
- reconstructed ODE evaluation.

These tests are intended to verify correct basic behavior of the package and
support reproducibility during development and refactoring.

Notes
-----
The tests in this file are designed as fast checks rather than exhaustive
validation of all numerical edge cases.

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

import numpy as np
from erfit import polyspace, dEntropy, MIDiscrete, MIKnn, getODEHandle


def test_polyspace_shape():
    X = np.array([[1., 2.], [3., 4.]])
    A, P = polyspace(X, 2)
    assert A.shape == (2, 6)
    assert P.shape == (6, 2)


def test_entropy_nonnegative():
    x = np.array([[1], [1], [2], [2]])
    assert dEntropy(x) >= 0
    assert MIDiscrete(x, x) >= 0


def test_getodehandle():
    P = np.array([[0, 0], [1, 0], [0, 1], [2, 0]])
    S = np.array([[1.0], [2.0], [3.0], [4.0]])
    f = getODEHandle(P, S)
    val = f(0.0, np.array([2.0, 5.0]))
    assert np.allclose(val, np.array([1 + 2*2 + 3*5 + 4*4]))
