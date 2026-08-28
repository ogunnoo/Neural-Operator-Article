"""Validate the cavity solver against the Ghia, Ghia & Shin (1982) benchmark.

The reference data below are the standard tabulated centreline velocities for
Re = 100 in a unit cavity driven by a *uniform* lid (u = 1 everywhere on the
top wall).  We run exactly that case -- no corner taper -- so the comparison is
apples to apples, then report the deviation.
"""

from __future__ import annotations

import numpy as np

from .cavity_solver import solve_cavity, grid

# Ghia et al. (1982), Re = 100.  u on the vertical centreline x = 0.5.
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                   -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                   0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

# v on the horizontal centreline y = 0.5.
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                   0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                   0.9609, 0.9688, 1.0000])
GHIA_V = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                   0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                   -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])


def run(n: int = 41, re: float = 100.0):
    x, y = grid(n)
    lid = np.ones(n)                       # uniform lid, matching the benchmark
    sol = solve_cavity(lid, re=re, n=n)

    mid = n // 2
    assert abs(x[mid] - 0.5) < 1e-12, "grid must contain x = 0.5"

    u_center = sol.u[:, mid]
    v_center = sol.v[mid, :]

    u_interp = np.interp(GHIA_Y, y, u_center)
    v_interp = np.interp(GHIA_X, x, v_center)

    u_err = np.max(np.abs(u_interp - GHIA_U))
    v_err = np.max(np.abs(v_interp - GHIA_V))
    u_rel = np.linalg.norm(u_interp - GHIA_U) / np.linalg.norm(GHIA_U)
    v_rel = np.linalg.norm(v_interp - GHIA_V) / np.linalg.norm(GHIA_V)

    return {
        "n": n,
        "re": re,
        "steps": sol.steps,
        "converged": sol.converged,
        "residual": sol.residual,
        "max_divergence": sol.max_divergence,
        "psi_min": float(sol.psi.min()),
        "u_max_abs_err_vs_ghia": float(u_err),
        "v_max_abs_err_vs_ghia": float(v_err),
        "u_rel_l2_vs_ghia": float(u_rel),
        "v_rel_l2_vs_ghia": float(v_rel),
    }


if __name__ == "__main__":
    import json
    import sys
    import time

    for n in (33, 41, 65):
        t0 = time.time()
        try:
            r = run(n=n)
            r["wall_time_s"] = round(time.time() - t0, 2)
            print(json.dumps(r, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"n={n} FAILED: {e}", file=sys.stderr)
