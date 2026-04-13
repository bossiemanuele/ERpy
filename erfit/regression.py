"""
Regression and Entropy-Based Sparse Model Selection
===================================================

This module implements the core regression and model-selection routines used in
the ERFit package for sparse identification of nonlinear dynamical systems.

Implemented functionality includes:
- least-squares fitting on selected supports,
- iteratively reweighted least squares for robust estimation,
- entropy-based forward selection,
- entropy-based backward elimination,
- complete ERFit sparse model identification.

The main entry point is `erfit`, which combines information-theoretic selection
with coefficient estimation to recover parsimonious dynamical models from a
candidate polynomial library.

Functions
---------
lsfit
    Fit coefficients by least squares on a prescribed support.
iterativeRWLS
    Perform robust iteratively reweighted least-squares fitting.
erforward
    Carry out entropy-based forward term selection.
erbackward
    Carry out entropy-based backward term elimination.
erfit
    Perform complete sparse model identification using ERFit.

Notes
-----
This module contains the main methodological contribution of the package and is
the primary algorithmic component used in nonlinear system discovery.

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
from .options import eroptset
from .entropy import tolEstimate


def _project(A, y):
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    if A.size == 0:
        return np.zeros_like(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def lsfit(A, Y, ix):
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    ix = np.asarray(ix)
    n = Y.shape[1]
    m = A.shape[1]
    S = np.zeros((m, n))
    for i in range(n):
        mask = ix[:, i].astype(bool) if ix.ndim == 2 else ix.astype(bool)
        if np.any(mask):
            coef, *_ = np.linalg.lstsq(A[:, mask], Y[:, i], rcond=None)
            S[mask, i] = coef
    return S[:, 0] if S.shape[1] == 1 else S


def iterativeRWLS(A, Y, ix, max_iter: int = 50, tol: float = 1e-10):
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    ix = np.asarray(ix)
    n = Y.shape[1]
    m = A.shape[1]
    S = np.zeros((m, n))
    for i in range(n):
        mask = ix[:, i].astype(bool) if ix.ndim == 2 else ix.astype(bool)
        X = A[:, mask]
        y = Y[:, i]
        if X.size == 0:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for _ in range(max_iter):
            r = y - X @ beta
            s = np.median(np.abs(r - np.median(r))) / 0.6745
            if s <= np.finfo(float).eps:
                break
            u = r / (s * 1.205)
            w = np.ones_like(u)
            nz = np.abs(u) > np.finfo(float).eps
            w[nz] = np.tanh(u[nz]) / u[nz]
            Wsqrt = np.sqrt(np.clip(w, 0.0, None))
            Xw = X * Wsqrt[:, None]
            yw = y * Wsqrt
            beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            if np.linalg.norm(beta_new - beta) <= tol * (1 + np.linalg.norm(beta)):
                beta = beta_new
                break
            beta = beta_new
        S[mask, i] = beta
    return S[:, 0] if S.shape[1] == 1 else S


def erforward(A, y, IXin, options):
    success = True
    ix = []
    IX = list(np.unique(np.concatenate([np.ravel(options.get('fkeepin', [])), np.ravel(options.get('keepin', []))])).astype(int))
    N = A.shape[1]
    options = dict(options)
    options['tol'], _ = tolEstimate(np.asarray(y).reshape(-1, 1), options)
    Imax = options['MIEstimator'](y, _project(A, y), options)
    Done = False
    IXin = np.asarray(IXin, dtype=int)
    while not Done:
        if len(ix) > 0:
            IX = IX + list(IXin[np.asarray(ix, dtype=int)])
        if len(IX) > 0:
            AIX = A[:, IX]
            Ilocal = options['MIEstimator'](y, _project(AIX, y))
        else:
            Ilocal = 0.0
        D = np.full(N, -np.inf, dtype=float)
        for i in range(N):
            if IXin[i] not in IX:
                cols = IX + [int(IXin[i])]
                f1 = _project(A[:, cols], y)
                f2 = _project(A[:, IX], y) if len(IX) > 0 else np.zeros_like(y)
                D[i] = options['MIEstimator'](y, f1, f2, options)
        val = np.max(D)
        ix = np.flatnonzero(D == val)
        ix = [int(ix[0])] if ix.size else []
        if ((Imax - Ilocal) <= options['tol']) or (val <= options['tol']):
            Done = True
        elif len(IX) > max(8, N / 2):
            success = False
            Done = True
    return np.array(IX, dtype=int), success


def erbackward(A, y, IX, options):
    val = -np.inf
    ix = []
    keepin = list(np.ravel(options.get('keepin', [])))
    IX = list(np.ravel(IX))
    while (val <= options['tol']) and (len(IX) > 1):
        if len(ix) > 0:
            del IX[ix[0]]
        D = np.full(len(IX), np.inf, dtype=float)
        for i in range(len(D)):
            if IX[i] not in keepin:
                rem = [j for j in IX if j != IX[i]]
                f1 = _project(A[:, IX], y)
                f2 = _project(A[:, rem], y) if len(rem) > 0 else np.zeros_like(y)
                D[i] = options['MIEstimator'](f1, y, f2, options)
        val = float(np.min(D))
        ix_arr = np.flatnonzero(D == val)
        ix = [int(ix_arr[0])] if ix_arr.size else []
    return np.array(IX, dtype=int)


def erfit(Phi, Xdot, options=None):
    Phi = np.asarray(Phi, dtype=float)
    Xdot = np.asarray(Xdot, dtype=float)
    assert Phi.shape[0] == Xdot.shape[0], 'All imputs should have the same number of rows.'
    if options is None:
        options = eroptset()
    else:
        options = dict(options)
    dim = Xdot.shape[1]
    m, N = Phi.shape
    S = np.zeros((N, dim))
    Info = {
        'estimatedTolerence': np.zeros(dim),
        'ForwardSelectedIndex': [None] * dim,
        'ForwardSuccess': np.zeros(dim, dtype=bool),
        'BackwardEliminatedIndex': [None] * dim,
        'Index': [None] * dim,
        'S': [None] * dim,
    }
    for i in range(dim):
        local_options = dict(options)
        local_options['tol'] = 0.0
        if not local_options.get('EmbeddedShuffleTest', False):
            In = np.zeros(local_options['numPerm'])
            for j in range(local_options['numPerm']):
                In[j] = local_options['MIEstimator'](Xdot[:, i], Xdot[np.random.permutation(m), i], local_options)
            local_options['tol'] = float(np.quantile(In, local_options['alpha']))
        Info['estimatedTolerence'][i] = local_options['tol']
        IX = np.arange(N)
        success = False
        if not local_options.get('skipForward', False):
            IX1, success = erforward(Phi, Xdot[:, i], IX, local_options)
        if (not success) or local_options.get('skipForward', False):
            IX1 = IX
        Info['ForwardSelectedIndex'][i] = IX1
        Info['ForwardSuccess'][i] = success
        IX2 = erbackward(Phi[:, IX1], Xdot[:, i], np.arange(len(IX1)), local_options)
        IXf = IX1[IX2]
        Info['BackwardEliminatedIndex'][i] = np.setdiff1d(IX1, IXf)
        if 0 not in IXf:
            IXf = np.concatenate([[0], IXf])
        index = np.zeros(S.shape[0], dtype=bool)
        index[IXf] = True
        S[:, i] = local_options['grayModelEstimator'](Phi, Xdot[:, i], index)
        if (abs(S[0, i]) < local_options['h']) and (np.count_nonzero(S[:, i]) > 1):
            S[0, i] = 0.0
            IXf = IXf[IXf != 0]
        Info['Index'][i] = IXf
        Info['S'][i] = S[IXf, i]
    return S, Info
