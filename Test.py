"""
End-to-End Demonstration and Validation Script for ERFit
=======================================================

This script provides a complete end-to-end demonstration of the ERFit workflow
for sparse identification of nonlinear dynamical systems. It serves both as a
functional validation of the package and as a reference implementation for
users.

The script performs the following steps:
1. Generate synthetic time-series data from benchmark dynamical systems.
2. Construct a polynomial feature library.
3. Estimate time derivatives from the simulated data.
4. Apply entropy-based forward and backward selection (ERFit).
5. Recover sparse governing equations.
6. Display and compare identified models with ground truth.

Purpose
-------
- Validate correctness of the ERFit pipeline.
- Provide a reproducible experimental setup.
- Serve as a template for applying ERFit to new systems.

Usage
-----
Run this script directly to execute all experiments:

    python test.py

Dependencies
------------
This script requires the ERFit package and standard scientific Python libraries
(e.g., NumPy, SciPy).

Notes
-----
This file is intended for experimentation and validation rather than unit
testing. For automated testing, refer to the `tests/` directory.

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

import numpy as np
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make sure Python can find the package
# ------------------------------------------------------------------
pkg_path = Path(__file__).resolve().parent / "ERpy"
sys.path.insert(0, str(pkg_path))

from erfit import getSystemDataset, erfit, eroptset
from erfit import lorenzODE, rosslerODE, vanderpol


# ------------------------------------------------------------------
# Utilities to convert sparse coefficients into readable equations
# ------------------------------------------------------------------
def monomial_to_string(powers, var_names):
    terms = []
    for v, p in zip(var_names, powers):
        if p == 0:
            continue
        elif p == 1:
            terms.append(v)
        else:
            terms.append(f"{v}^{int(p)}")
    return "1" if len(terms) == 0 else "*".join(terms)


def sparse_system_to_equations(S, P, state_names, deriv_names=None, tol=1e-10):
    """
    Convert sparse coefficient matrix S and power matrix P into readable equations.

    Parameters
    ----------
    S : ndarray, shape (#library_terms, #states)
        Sparse coefficient matrix.
    P : ndarray, shape (#library_terms, #states)
        Power/exponent matrix describing the monomials.
    state_names : list[str]
        Names of the state variables.
    deriv_names : list[str], optional
        Names of the derivatives.
    tol : float
        Coefficients with absolute value <= tol are treated as zero.
    """
    if deriv_names is None:
        deriv_names = [f"d{v}/dt" for v in state_names]

    eqs = []
    for j in range(S.shape[1]):
        parts = []
        for i in range(S.shape[0]):
            c = S[i, j]
            if abs(c) <= tol:
                continue

            mono = monomial_to_string(P[i], state_names)

            if mono == "1":
                parts.append(f"{c:.8g}")
            else:
                parts.append(f"{c:.8g}*{mono}")

        rhs = " + ".join(parts).replace("+ -", "- ")
        if rhs.strip() == "":
            rhs = "0"
        eqs.append(f"{deriv_names[j]} = {rhs}")
    return eqs


def print_recovered_system(title, S, data, state_names, true_equations, info=None):
    """
    Pretty-print recovered coefficient matrix, equations, and selected term indices.
    """
    print(f"\n{title}")
    print("-" * len(title))

    print("\nRecovered sparse coefficient matrix S:")
    print(S)

    print("\nRecovered equations:")
    recovered_eqs = sparse_system_to_equations(
        S, data["PowrMatrix"], state_names=state_names
    )
    for eq in recovered_eqs:
        print("  " + eq)

    print("\nTrue equations:")
    for eq in true_equations:
        print("  " + eq)

    if info is not None and "Index" in info:
        print("\nSelected library terms by equation:")
        for k, idx in enumerate(info["Index"]):
            print(f"  Equation {k+1}: term indices {idx}")


def run_example(
    name,
    odefun,
    dim,
    state_names,
    true_equations,
    seed=1,
    sample_size=1000,
    tao=0.01,
    expan_order=4,
    eps1=0.005,
    eps2=0.0,
    berp=0.0,
):
    print("\n" + "=" * 80)
    print(f"{name.upper()} EXAMPLE")
    print("=" * 80)

    np.random.seed(seed)

    data = getSystemDataset(
        odefun,
        dim=dim,
        SampleSize=sample_size,
        tao=tao,
        expanOrder=expan_order,
        eps1=eps1,
        eps2=eps2,
        Berp=berp,
        skipTrans=True,
    )

    opts = eroptset()

    S_er, info_er = erfit(data["Phi"], data["Xdot"], opts)

    print_recovered_system(
        title="Plain ER result",
        S=S_er,
        data=data,
        state_names=state_names,
        true_equations=true_equations,
        info=info_er,
    )

    return {
        "data": data,
        "S_er": S_er,
        "info_er": info_er,
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Lorenz system
    # ------------------------------------------------------------------
    lorenz_true = [
        "dx/dt = -10*x + 10*y",
        "dy/dt = 28*x - y - x*z",
        "dz/dt = x*y - (8/3)*z",
    ]

    run_example(
        name="Lorenz",
        odefun=lorenzODE,
        dim=3,
        state_names=["x", "y", "z"],
        true_equations=lorenz_true,
        seed=1,
        sample_size=1000,
        tao=0.01,
        expan_order=4,
        eps1=0.005,
        eps2=0.0,
        berp=0.0,
    )

    # ------------------------------------------------------------------
    # 2) Rössler system
    # ------------------------------------------------------------------
    rossler_true = [
        "dx/dt = -y - z",
        "dy/dt = x + 0.2*y",
        "dz/dt = 0.2 + x*z - 5.7*z",
    ]

    run_example(
        name="Rossler",
        odefun=rosslerODE,
        dim=3,
        state_names=["x", "y", "z"],
        true_equations=rossler_true,
        seed=1,
        sample_size=1000,
        tao=0.01,
        expan_order=4,
        eps1=0.005,
        eps2=0.0,
        berp=0.0,
    )

    # ------------------------------------------------------------------
    # 3) Van der Pol oscillator
    # ------------------------------------------------------------------
    vdp_true = [
        "dx/dt = y",
        "dy/dt = 5*y - 5*x^2*y - x",
    ]

    run_example(
        name="Van der Pol",
        odefun=vanderpol,
        dim=2,
        state_names=["x", "y"],
        true_equations=vdp_true,
        seed=1,
        sample_size=1000,
        tao=0.01,
        expan_order=4,
        eps1=0.005,
        eps2=0.0,
        berp=0.0,
    )