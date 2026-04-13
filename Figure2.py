"""
Comparison of Mutual Information Estimators in Entropic Regression
=================================================================

This script compares the performance of Entropic Regression (ER) using two
different mutual information (MI) estimators:

- Binning-based estimator (discrete histogram)
- k-nearest neighbor estimator (kNN)

The comparison is conducted on the Lorenz system across a range of measurement
sizes.

Experimental protocol:
- 100 independent runs per number of measurements
- no outliers
- step size = 0.0005
- Gaussian measurement noise epsilon = 1e-4
- 5th-order polynomial expansion
- error = ||a_true - a_est||_2
- summary statistic: median error over runs

The objective is to assess how the choice of MI estimator influences the
performance of ER.

Author and Affiliation
----------------------
Emanuele Bossi, Embry-Riddle Aeronautical University

Date Last Modified
------------------
04/05/2026

License
-------
MIT

Citation
--------
Bossi, E. (2026) "ERpy: Python Implementation of Entropic Regression for Nonlinear System Identification."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Package path
# ---------------------------------------------------------------------
PKG_PATH = Path(__file__).resolve().parent / "ERpy"
sys.path.insert(0, str(PKG_PATH))

from erfit import getSystemDataset, erfit, eroptset, lorenzODE
from erfit.entropy import MIDiscrete, MIKnn


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
N_MEAS = np.linspace(200, 5000, 20).astype(int)
N_RUNS = 100

DT = 0.0005
EPS1 = 1e-4
EPS2 = 0.0
BERP = 0.0
EXPAN_ORDER = 5
SEED = 1

OUTDIR = Path(__file__).resolve().parent / "compare_mi_estimators"
OUTDIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_2d(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    return S


def find_term_index(power_matrix: np.ndarray, powers: tuple[int, int, int]) -> int:
    for i, row in enumerate(power_matrix):
        if tuple(int(v) for v in row) == powers:
            return i
    raise ValueError(f"Monomial {powers} not found in power matrix.")


def build_true_lorenz_coefficients(power_matrix: np.ndarray) -> np.ndarray:
    n_terms = power_matrix.shape[0]
    S_true = np.zeros((n_terms, 3), dtype=float)

    idx_x  = find_term_index(power_matrix, (1, 0, 0))
    idx_y  = find_term_index(power_matrix, (0, 1, 0))
    idx_z  = find_term_index(power_matrix, (0, 0, 1))
    idx_xy = find_term_index(power_matrix, (1, 1, 0))
    idx_xz = find_term_index(power_matrix, (1, 0, 1))

    S_true[idx_x, 0] = -10.0
    S_true[idx_y, 0] = 10.0

    S_true[idx_x, 1]  = 28.0
    S_true[idx_y, 1]  = -1.0
    S_true[idx_xz, 1] = -1.0

    S_true[idx_xy, 2] = 1.0
    S_true[idx_z, 2]  = -(8.0 / 3.0)

    return S_true


def coefficient_error(S_est: np.ndarray, S_true: np.ndarray) -> float:
    return float(np.linalg.norm((ensure_2d(S_est) - ensure_2d(S_true)).ravel(), ord=2))


def support_size(S: np.ndarray, tol: float = 1e-12) -> int:
    return int(np.sum(np.abs(ensure_2d(S)) > tol))


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
errors_bin = []
errors_knn = []

supports_bin = []
supports_knn = []

for n in N_MEAS:
    print(f"\nRunning N = {n}")

    run_errors_bin = []
    run_errors_knn = []

    run_supports_bin = []
    run_supports_knn = []

    for k in range(N_RUNS):
        seed = SEED + k + 1000 * n
        np.random.seed(seed)

        data = getSystemDataset(
            lorenzODE,
            dim=3,
            SampleSize=n,
            tao=DT,
            expanOrder=EXPAN_ORDER,
            eps1=EPS1,
            eps2=EPS2,
            Berp=BERP,
            skipTrans=True,
        )

        Phi = np.asarray(data["Phi"], dtype=float)
        Xdot = np.asarray(data["Xdot"], dtype=float)
        P = np.asarray(data["PowrMatrix"], dtype=int)

        S_true = build_true_lorenz_coefficients(P)

        # ---- Binning ----
        opts_bin = eroptset(
            MIEstimator=MIDiscrete,
            BinMethod="fixed",
            numBins=16,
        )
        S_bin, _ = erfit(Phi, Xdot, opts_bin)
        run_errors_bin.append(coefficient_error(S_bin, S_true))
        run_supports_bin.append(support_size(S_bin))

        # ---- kNN ----
        opts_knn = eroptset(
            MIEstimator=MIKnn,
            K=2,
        )
        S_knn, _ = erfit(Phi, Xdot, opts_knn)
        run_errors_knn.append(coefficient_error(S_knn, S_true))
        run_supports_knn.append(support_size(S_knn))

    # ---- Median aggregation ----
    errors_bin.append(np.median(run_errors_bin))
    errors_knn.append(np.median(run_errors_knn))

    supports_bin.append(int(np.median(run_supports_bin)))
    supports_knn.append(int(np.median(run_supports_knn)))

    print(f"  Binning MI error (median): {errors_bin[-1]:.6e}")
    print(f"  kNN MI error (median)    : {errors_knn[-1]:.6e}")


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
csv_path = OUTDIR / "compare_binning_vs_knn.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("n_measurements,error_binning,error_knn,support_binning,support_knn\n")
    for i, n in enumerate(N_MEAS):
        f.write(
            f"{n},{errors_bin[i]:.12e},{errors_knn[i]:.12e},{supports_bin[i]},{supports_knn[i]}\n"
        )

print(f"\nSaved results to:\n{csv_path}")


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
})

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(
    N_MEAS,
    errors_bin,
    marker="o",
    markerfacecolor="white",
    markeredgewidth=1.6,
    label="ER (binning MI)",
)

ax.plot(
    N_MEAS,
    errors_knn,
    marker="s",
    markerfacecolor="white",
    markeredgewidth=1.6,
    label="ER (kNN MI)",
)

ax.set_yscale("log")
ax.set_xlabel("Number of Measurements")
ax.set_ylabel(r"$(error)$")
ax.set_title("Lorenz system: ER with binning MI vs kNN MI")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()

png_path = OUTDIR / "compare_binning_vs_knn.png"
pdf_path = OUTDIR / "compare_binning_vs_knn.pdf"

fig.savefig(png_path, dpi=400)
fig.savefig(pdf_path)
plt.close(fig)

print(f"Saved plot to:\n{png_path}\n{pdf_path}")