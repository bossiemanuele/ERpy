"""
Entropy and Mutual Information Estimators
=========================================

This module implements the information-theoretic quantities used by ERFit for
feature selection and stopping criteria in sparse nonlinear system
identification.

Implemented functionality includes:
- discrete Shannon entropy estimation,
- histogram-based discretization of continuous variables,
- discrete mutual information (MI),
- conditional mutual information (CMI),
- k-nearest-neighbor MI estimation,
- tolerance estimation based on information measures.

These estimators are used to evaluate the relevance of candidate library terms
during entropy-based forward and backward selection.

Functions
---------
dEntropy
    Compute discrete Shannon entropy.
discretizeData
    Discretize continuous data using histogram-based binning.
MIDiscrete
    Compute discrete mutual information or conditional mutual information.
MIKnn
    Estimate mutual information using a k-nearest-neighbor method.
tolEstimate
    Estimate a numerical tolerance for ERFit stopping decisions.

Notes
-----
This module is central to the ERFit methodology, where information gain is used
to quantify the contribution of candidate terms beyond standard regression error
metrics.

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

import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import digamma
from .options import eroptset


def dEntropy(X: np.ndarray) -> float:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    _, counts = np.unique(X, axis=0, return_counts=True)
    P = counts / counts.sum()
    return float(-np.dot(P, np.log2(P)))


def _hist_edges_like_matlab(X: np.ndarray, options: dict):
    X = np.asarray(X, dtype=float)
    flat = X.ravel()
    method = options.get("BinMethod", "fixed")
    if method == "fixed":
        return np.histogram_bin_edges(flat, bins=int(options.get("numBins", 16)))
    if method == "integers":
        mn = math.floor(np.nanmin(flat))
        mx = math.ceil(np.nanmax(flat))
        return np.arange(mn - 0.5, mx + 1.5, 1.0)
    return np.histogram_bin_edges(flat, bins=method)


def discretizeData(X: np.ndarray, options: dict | None = None):
    if options is None:
        options = eroptset()
    X = np.asarray(X)
    edges = _hist_edges_like_matlab(X, options)
    symbols = np.digitize(X, edges[1:-1], right=False) + 1
    outside = (X < edges[0]) | (X > edges[-1])
    symbols = symbols.astype(float)
    symbols[outside] = np.nan
    return symbols, edges


def _maybe_options(options):
    return eroptset() if options is None else options


def MIDiscrete(x, y, *args):
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[0] == y.shape[0], 'All imputs should have the same number of rows.'

    options = eroptset()
    if args and isinstance(args[-1], dict):
        options = args[-1]
        args = args[:-1]
    Ni = len(args)
    assert Ni in (0, 1), 'Error01: Number of inmputs'
    N = x.shape[0]

    x, _ = discretizeData(x, options)
    y, _ = discretizeData(y, options)

    def miDiscrete(x, y, options):
        return dEntropy(x) + dEntropy(y) - dEntropy(np.column_stack([x, y]))

    def cmiDiscrete(x, y, z, options):
        return dEntropy(np.column_stack([x, z])) + dEntropy(np.column_stack([y, z])) - dEntropy(z) - dEntropy(np.column_stack([x, y, z]))

    if Ni == 0:
        I = miDiscrete(x, y, options)
        if options.get("EmbeddedShuffleTest", False):
            In = np.zeros(options["numPerm"])
            for i in range(options["numPerm"]):
                In[i] = miDiscrete(x, y[np.random.permutation(N)], options)
            In.sort()
            tol = In[int(math.floor(options["alpha"] * options["numPerm"])) - 1]
            if I < tol:
                I = 0.0
        return float(I)
    else:
        z = np.asarray(args[0])
        z, _ = discretizeData(z, options)
        assert x.shape[0] == z.shape[0], 'All imputs should have the same number of rows.'
        I = cmiDiscrete(x, y, z, options)
        if options.get("EmbeddedShuffleTest", False):
            In = np.zeros(options["numPerm"])
            for i in range(options["numPerm"]):
                In[i] = cmiDiscrete(x, y[np.random.permutation(N)], z, options)
            In.sort()
            tol = In[int(math.floor(options["alpha"] * options["numPerm"])) - 1]
            if I < tol:
                I = 0.0
        return float(I)


def _pairwise_minkowski(X, Y, p):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    metric = 'chebyshev' if math.isinf(p) else 'minkowski'
    kwargs = {} if math.isinf(p) else {'p': p}
    return cdist(X, Y, metric=metric, **kwargs)


def _reshape2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _knn_eps(JS, K, p):
    JS = _reshape2d(JS)
    tree = cKDTree(JS)
    d, _ = tree.query(JS, k=K + 1, p=p)
    return d[:, K]


def _count_strict_within(X, eps, p):
    X = _reshape2d(X)
    tree = cKDTree(X)
    radii = np.nextafter(eps, -np.inf)
    out = np.empty(X.shape[0], dtype=int)
    for i, (pt, r) in enumerate(zip(X, radii)):
        out[i] = len(tree.query_ball_point(pt, r=float(r), p=p)) - 1
    return out


def MIKnn(x, y, *args):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape[0] == y.shape[0], 'All imputs should have the same number of rows.'

    options = eroptset()
    if args and isinstance(args[-1], dict):
        options = args[-1]
        args = args[:-1]
    Ni = len(args)
    assert Ni in (0, 1), 'Error01: Number of inmputs'
    N = x.shape[0]

    def miKnn(x, y, options):
        K = int(options['K'])
        p = options['p']
        JS = np.column_stack([x, y])
        n = JS.shape[0]
        p_tree = np.inf if math.isinf(p) else p
        epsilon = _knn_eps(JS, K, p_tree)
        nx = _count_strict_within(x, epsilon, p_tree)
        ny = _count_strict_within(y, epsilon, p_tree)
        return float(digamma(K) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))

    def cmiKnn(x, y, z, options):
        K = int(options['K'])
        p = options['p']
        if z.size == 0:
            return miKnn(x, y, options)
        JS = np.column_stack([x, y, z])
        p_tree = np.inf if math.isinf(p) else p
        epsilon = _knn_eps(JS, K, p_tree)
        nxz = _count_strict_within(np.column_stack([x, z]), epsilon, p_tree)
        nyz = _count_strict_within(np.column_stack([y, z]), epsilon, p_tree)
        nz = _count_strict_within(z, epsilon, p_tree)
        return float(digamma(K) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1)))

    if Ni == 0:
        I = miKnn(x, y, options)
        if options.get("EmbeddedShuffleTest", False):
            In = np.zeros(options["numPerm"])
            for i in range(options["numPerm"]):
                In[i] = miKnn(x, y[np.random.permutation(N)], options)
            In.sort()
            tol = In[int(math.floor(options["alpha"] * options["numPerm"])) - 1]
            if I < tol:
                I = 0.0
        return float(I)
    else:
        z = np.asarray(args[0], dtype=float)
        assert x.shape[0] == z.shape[0], 'All imputs should have the same number of rows.'
        I = cmiKnn(x, y, z, options)
        if options.get("EmbeddedShuffleTest", False):
            In = np.zeros(options["numPerm"])
            for i in range(options["numPerm"]):
                In[i] = cmiKnn(x, y[np.random.permutation(N)], z, options)
            In.sort()
            tol = In[int(math.floor(options["alpha"] * options["numPerm"])) - 1]
            if I < tol:
                I = 0.0
        return float(I)


def tolEstimate(y, options):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    N = y.shape[0]
    In = np.zeros(options['numPerm'])
    for i in range(options['numPerm']):
        In[i] = options['MIEstimator'](y, y[np.random.permutation(N), :], options)
    tol = float(np.quantile(In, options['alpha']))
    return tol, In
