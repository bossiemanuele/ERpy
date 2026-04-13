"""
Synthetic Data Generation and Numerical Differentiation
======================================================

This module provides utilities for generating benchmark dynamical-system data
for entropy-based sparse regression experiments. It includes:

- central finite-difference estimation of state derivatives,
- built-in benchmark ODEs (Lorenz, Rössler, and Van der Pol),
- numerical integration of continuous-time systems,
- corruption of trajectories with Gaussian and burst noise,
- construction of polynomial feature libraries for system identification.

The main entry point is `getSystemDataset`, which simulates a nonlinear ODE,
adds optional measurement corruption, estimates derivatives, and returns the
feature matrix and metadata needed for ERFit-based model recovery.

Functions
---------
centralDifference
    Estimate time derivatives using a centered finite-difference scheme.
lorenzODE
    Lorenz attractor benchmark system.
rosslerODE
    Rössler attractor benchmark system.
vanderpol
    Van der Pol oscillator benchmark system.
getSystemDataset
    Generate a complete identification dataset from a user-specified ODE.

Notes
-----
The generated dataset is intended for sparse nonlinear system identification
and benchmarking of entropy-based forward-backward selection methods.

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
from scipy.integrate import solve_ivp
from .poly import polyspace


def centralDifference(X, h):
    X = np.asarray(X, dtype=float)
    Xdot = (1.0 / (2.0 * h)) * (X[2:, :] - X[:-2, :])
    Xtrim = X[1:-1, :].copy()
    return Xdot, Xtrim


def lorenzODE(t, x):
    x = np.asarray(x, dtype=float)
    xdot = np.zeros_like(x, dtype=float)
    xdot[0] = 10.0 * (x[1] - x[0])
    xdot[1] = 28.0 * x[0] - x[0] * x[2] - x[1]
    xdot[2] = x[0] * x[1] - (8.0 / 3.0) * x[2]
    return xdot


def rosslerODE(t, x):
    x = np.asarray(x, dtype=float)
    return np.array([-x[1] - x[2], x[0] + 0.2 * x[1], 0.2 + x[2] * (x[0] - 5.7)], dtype=float)


def vanderpol(t, x):
    x = np.asarray(x, dtype=float)
    return np.array([x[1], 5.0 * (1.0 - x[0] ** 2) * x[1] - x[0]], dtype=float)


def _ode45_like(odefun, t_eval, x0):
    sol = solve_ivp(odefun, (float(t_eval[0]), float(t_eval[-1])), np.asarray(x0, dtype=float), t_eval=t_eval, method='RK45', rtol=1e-3, atol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def getSystemDataset(odefun, **kwargs):
    options = {
        'odeSolver': _ode45_like,
        'SampleSize': 1000,
        'derivativeEstimator': centralDifference,
        'dim': 3,
        'tao': 0.01,
        'eps1': 0.005,
        'eps2': 0.0,
        'Berp': 0.0,
        'expanOrder': 4,
        'initCondition': 'rand',
        'skipTrans': True,
    }
    options.update(kwargs)
    if isinstance(options['initCondition'], (list, tuple, np.ndarray)):
        x0 = np.asarray(options['initCondition'], dtype=float)
    else:
        x0 = np.random.rand(int(options['dim']))
    Info = {'x0': x0.copy()}
    h = float(options['tao'])
    N = int(options['SampleSize'])
    X = np.empty((0, len(x0)))
    if options['skipTrans']:
        _, X = options['odeSolver'](odefun, np.array([0.0, 100.0]), x0)
    if X.size:
        x0 = X[-1, :].copy()
    Info['x0'] = x0.copy()
    t_eval = np.arange(0.0, (N + 2) * h + 1e-15, h)
    t, X = options['odeSolver'](odefun, t_eval, x0)
    Info['Xclean'] = X.copy()
    Info['t'] = t.copy()
    X = X + float(options['eps1']) * np.random.randn(*X.shape)
    Info['Xbasenoise'] = X.copy()
    IX = (np.random.rand(X.shape[0]) <= float(options['Berp']))
    if np.any(IX):
        X[IX, :] = X[IX, :] + float(options['eps2']) * np.random.randn(np.sum(IX), X.shape[1])
    Info['Xcorrupted'] = X.copy()
    Info['CorruptionIndex'] = IX.copy()
    Xdot, Xtrim = options['derivativeEstimator'](X, h)
    Info['Xdot'] = Xdot.copy()
    Phi, P = polyspace(Xtrim, int(options['expanOrder']))
    Info['Phi'] = Phi
    Info['PowrMatrix'] = P
    Info['h'] = h
    Info['eps1'] = options['eps1']
    Info['eps2'] = options['eps2']
    Info['Berp'] = options['Berp']
    Info['ExpansionOrder'] = options['expanOrder']
    return Info
