"""
Recreate Fig. 2 (Lorenz system) from AlMomani et al. (2020)
===========================================================

This script reproduces the Lorenz-system comparison protocol used for Fig. 2 in AlMomani et al. (2020):
- 10 independent runs (Note: AlMomani et. al used 100 independent runs)
- no outliers
- step size = 0.0005
- Gaussian measurement noise epsilon = 1e-4
- 5th-order polynomial expansion
- error = ||a_true - a_est||_2
- summary statistic = median error over runs

Implemented methods:
- LS
- LASSO
- SINDy (STLSQ-style hard thresholding)
- ER

Author and Affiliation
----------------------
Emanuele Bossi, Embry-Riddle Aeronautical University

Date Last Modified
------------------
04/04/2026

License
-------
MIT

Citation
--------
Bossi, E. (2026) "ERpy: Python Implementation of Entropic Regression for Nonlinear System Identification."
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------
# Make sure Python can find the ER package
# ---------------------------------------------------------------------
PKG_PATH = Path(__file__).resolve().parent / "ERpy"
sys.path.insert(0, str(PKG_PATH))

from erfit import getSystemDataset, erfit, eroptset, lorenzODE


# ---------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------
N_RUNS = 10                # number of independent runs
DT = 0.0005                # step size used in Fig. 2
EPS1 = 1e-4                # Gaussian noise level
EPS2 = 0.0                 # no outlier burst noise
BERP = 0.0                 # no outliers
EXPAN_ORDER = 5            # 5th-order polynomial expansion
MEASUREMENT_COUNTS = np.linspace(200, 5000, 20).astype(int)

# Plot output
OUTDIR = Path(__file__).resolve().parent / "fig2_reproduction"
OUTDIR.mkdir(exist_ok=True)

# Reproducibility
BASE_SEED = 20260405


# ---------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------
def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


def lorenz_true_coefficients(power_matrix: np.ndarray) -> np.ndarray:
    """
    Build the true sparse coefficient matrix S_true corresponding to the Lorenz system:
        xdot = -10 x + 10 y
        ydot =  28 x - y - x z
        zdot =  x y - (8/3) z
    in the polynomial library encoded by power_matrix.
    """
    power_matrix = np.asarray(power_matrix, dtype=int)
    n_terms = power_matrix.shape[0]
    S_true = np.zeros((n_terms, 3), dtype=float)

    def find_term(powers: tuple[int, int, int]) -> int:
        for idx, row in enumerate(power_matrix):
            if tuple(int(v) for v in row.tolist()) == powers:
                return idx
        raise ValueError(f"Monomial {powers} not found in PowrMatrix.")

    idx_x = find_term((1, 0, 0))
    idx_y = find_term((0, 1, 0))
    idx_z = find_term((0, 0, 1))
    idx_xy = find_term((1, 1, 0))
    idx_xz = find_term((1, 0, 1))

    # xdot
    S_true[idx_x, 0] = -10.0
    S_true[idx_y, 0] = 10.0

    # ydot
    S_true[idx_x, 1] = 28.0
    S_true[idx_y, 1] = -1.0
    S_true[idx_xz, 1] = -1.0

    # zdot
    S_true[idx_xy, 2] = 1.0
    S_true[idx_z, 2] = -(8.0 / 3.0)

    return S_true


def flatten_error(S_est: np.ndarray, S_true: np.ndarray) -> float:
    """
    Parameter error = ||a_true - a_est||_2, with all equations flattened together.
    """
    S_est = ensure_2d(S_est)
    S_true = ensure_2d(S_true)
    return float(np.linalg.norm((S_est - S_true).ravel(), ord=2))


def least_squares_fit(Phi: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    """
    Dense least-squares fit for all state equations.
    """
    Phi = np.asarray(Phi, dtype=float)
    Xdot = ensure_2d(Xdot)
    coef, *_ = np.linalg.lstsq(Phi, Xdot, rcond=None)
    return coef


def lasso_fit(Phi: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    """
    LASSO with 10 log-spaced alphas and 5-fold CV, matching the manuscript description.
    """
    Phi = np.asarray(Phi, dtype=float)
    Xdot = ensure_2d(Xdot)

    n_terms = Phi.shape[1]
    n_targets = Xdot.shape[1]
    coef = np.zeros((n_terms, n_targets), dtype=float)

    # 10 values on a log scale; sklearn takes alphas in descending order
    alphas = np.logspace(-6, 0, 10)[::-1]
    cv = KFold(n_splits=5, shuffle=True, random_state=BASE_SEED)

    for j in range(n_targets):
        model = LassoCV(
            alphas=alphas,
            cv=cv,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            random_state=BASE_SEED,
        )
        model.fit(Phi, Xdot[:, j])
        coef[:, j] = model.coef_

    return coef

def sindy_stlsq_fit(
    Phi: np.ndarray,
    Xdot: np.ndarray,
    lam: float = 0.02,
    max_iter: int = 25
) -> np.ndarray:
    """
    Sequential thresholded least squares (STLSQ), used here as the SINDy solver.
    For Lorenz, the manuscript states lambda = 0.02.
    """
    Phi = np.asarray(Phi, dtype=float)
    Xdot = ensure_2d(Xdot)

    n_terms = Phi.shape[1]
    n_targets = Xdot.shape[1]
    coef = np.zeros((n_terms, n_targets), dtype=float)

    for j in range(n_targets):
        y = Xdot[:, j]
        c, *_ = np.linalg.lstsq(Phi, y, rcond=None)

        for _ in range(max_iter):
            small = np.abs(c) < lam
            c[small] = 0.0
            active = ~small

            if not np.any(active):
                break

            c_new = np.zeros_like(c)
            c_active, *_ = np.linalg.lstsq(Phi[:, active], y, rcond=None)
            c_new[active] = c_active

            if np.allclose(c, c_new, atol=1e-12, rtol=1e-10):
                c = c_new
                break

            c = c_new

        coef[:, j] = c

    return coef


def er_fit(Phi: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
    """
    Entropic Regression using your package.
    """
    opts = eroptset()
    S, _ = erfit(Phi, Xdot, opts)
    return ensure_2d(S)


def generate_dataset(n_meas: int, seed: int) -> dict:
    """
    Generate one noisy Lorenz dataset using your package.
    """
    np.random.seed(seed)
    data = getSystemDataset(
        lorenzODE,
        dim=3,
        SampleSize=int(n_meas),
        tao=DT,
        expanOrder=EXPAN_ORDER,
        eps1=EPS1,
        eps2=EPS2,
        Berp=BERP,
        skipTrans=True,
    )
    return data


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main() -> None:
    set_pub_style()

    methods = {
        "LS": least_squares_fit,
        "LASSO": lasso_fit,
        "SINDy": sindy_stlsq_fit,
        "ER": er_fit,
    }

    median_errors: dict[str, list[float]] = {name: [] for name in methods.keys()}

    for n_meas in MEASUREMENT_COUNTS:
        run_errors: dict[str, list[float]] = {name: [] for name in methods.keys()}

        print(f"\nNumber of measurements = {n_meas}")

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + 10000 * int(n_meas) + run_idx
            data = generate_dataset(n_meas=n_meas, seed=seed)

            Phi = np.asarray(data["Phi"], dtype=float)
            Xdot = ensure_2d(np.asarray(data["Xdot"], dtype=float))
            P = np.asarray(data["PowrMatrix"], dtype=int)

            S_true = lorenz_true_coefficients(P)

            for name, fit_fn in methods.items():
                try:
                    S_est = fit_fn(Phi, Xdot)
                    err = flatten_error(S_est, S_true)
                except Exception:
                    err = np.nan
                run_errors[name].append(err)

            if (run_idx + 1) % 10 == 0:
                print(f"  completed run {run_idx + 1}/{N_RUNS}")

        for name in methods.keys():
            vals = np.asarray(run_errors[name], dtype=float)
            vals = vals[np.isfinite(vals)]
            median_errors[name].append(float(np.median(vals)) if vals.size > 0 else np.nan)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 12))

    style = {
        "LS":    dict(marker="s", fillstyle="none"),
        "LASSO": dict(marker="o", fillstyle="none"),
        "SINDy": dict(marker="D", fillstyle="none"),
        "ER":    dict(marker="*", fillstyle="none"),
    }

    for name in ["LS", "LASSO", "SINDy", "ER"]:
        ax.plot(
            MEASUREMENT_COUNTS,
            np.asarray(median_errors[name], dtype=float),
            label=name,
            **style[name]
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Measurements")
    ax.set_ylabel(r"$(error)$")
    ax.set_title("Lorenz system")
    ax.grid(True, which="both")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper right")

    # Match the visual feel of the paper figure
    ax.set_xlim(MEASUREMENT_COUNTS.min() - 50, MEASUREMENT_COUNTS.max() + 50)
    ax.set_ylim(1e-4, 1e6)

    fig.savefig(OUTDIR / "fig2_lorenz_reproduction.png")
    fig.savefig(OUTDIR / "fig2_lorenz_reproduction.pdf")
    plt.close(fig)

    print("\nSaved:")
    print(OUTDIR / "fig2_lorenz_reproduction.png")
    print(OUTDIR / "fig2_lorenz_reproduction.pdf")

    # Also save raw numeric results
    out_txt = OUTDIR / "fig2_lorenz_reproduction_results.csv"
    with open(out_txt, "w", encoding="utf-8") as f:
        header = ["n_measurements"] + list(methods.keys())
        f.write(",".join(header) + "\n")
        for i, n_meas in enumerate(MEASUREMENT_COUNTS):
            row = [str(int(n_meas))] + [f"{median_errors[name][i]:.12e}" for name in methods.keys()]
            f.write(",".join(row) + "\n")

    print(out_txt)


if __name__ == "__main__":
    main()