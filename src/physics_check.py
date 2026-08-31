"""Does the *purely supervised* operator respect an unenforced physical law?

The DeepONet has no PDE residual and no hard-coded boundary conditions.
Boundary nodes are nevertheless present in its supervised targets, whereas
incompressibility is neither imposed nor directly penalised. This script
separates those two cases for the article's "Where did the physics go?" section.
"""

from __future__ import annotations

import json

import numpy as np

from .config import CFG, DATADIR
from .dataset import generate
from .evaluate import load_model, predict_fields


def main(verbose: bool = True):
    data = generate(verbose=False)
    x, y = data["x"], data["y"]
    h = 1.0 / (CFG.n - 1)
    fields_fn, _ = load_model()

    rows = []
    for i, g in enumerate(data["test_g"]):
        up, vp = predict_fields(fields_fn, g, x, y)
        ut, vt = data["test_u"][i], data["test_v"][i]

        # Discrete divergence on the interior, same stencil used for the solver.
        def div(u, v):
            return ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
                    + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h))

        # With cavity length L=1, division by this speed scale makes the
        # divergence statistic dimensionless.
        scale = float(np.sqrt(ut**2 + vt**2).max())
        wall_speed = np.sqrt(up**2 + vp**2)
        wall_max_speed = float(max(
            wall_speed[0, :].max(),
            wall_speed[:, 0].max(),
            wall_speed[:, -1].max(),
        ))

        rows.append({
            "test_index": i + 1,
            "pred_max_abs_div": float(np.abs(div(up, vp)).max()),
            "pred_rms_div": float(np.sqrt((div(up, vp) ** 2).mean())),
            "ref_max_abs_div": float(np.abs(div(ut, vt)).max()),
            # No symbolic no-slip constraint was imposed, but these boundary
            # nodes were included in the supervised full-grid targets.
            "wall_max_abs_u": float(max(np.abs(up[0, :]).max(),
                                        np.abs(up[:, 0]).max(),
                                        np.abs(up[:, -1]).max())),
            "wall_max_abs_v": float(max(np.abs(vp[0, :]).max(),
                                        np.abs(vp[:, 0]).max(),
                                        np.abs(vp[:, -1]).max())),
            "wall_max_speed": wall_max_speed,
            # Lid: does the prediction reproduce the imposed g(x)?
            "lid_max_abs_err": float(np.abs(up[-1, :] - g).max()),
            "speed_scale": scale,
        })
        if verbose:
            r = rows[-1]
            print(f"Test {i+1}: |div u|max pred {r['pred_max_abs_div']:.3e} "
                  f"(ref {r['ref_max_abs_div']:.1e})   "
                  f"wall |u|max {r['wall_max_abs_u']:.4f}   "
                  f"lid err {r['lid_max_abs_err']:.4f}")

    summary = {
        "per_test_function": rows,
        "mean_pred_max_abs_div": float(np.mean([r["pred_max_abs_div"] for r in rows])),
        "mean_pred_rms_div": float(np.mean([r["pred_rms_div"] for r in rows])),
        "mean_ref_max_abs_div": float(np.mean([r["ref_max_abs_div"] for r in rows])),
        "mean_wall_max_abs_u": float(np.mean([r["wall_max_abs_u"] for r in rows])),
        "mean_wall_max_abs_v": float(np.mean([r["wall_max_abs_v"] for r in rows])),
        "mean_wall_max_speed": float(np.mean([r["wall_max_speed"] for r in rows])),
        "mean_lid_max_abs_err": float(np.mean([r["lid_max_abs_err"] for r in rows])),
        "mean_speed_scale": float(np.mean([r["speed_scale"] for r in rows])),
    }
    summary["div_relative_to_speed_scale"] = (
        summary["mean_pred_max_abs_div"] / summary["mean_speed_scale"])
    summary["rms_div_relative_to_speed_scale"] = (
        summary["mean_pred_rms_div"] / summary["mean_speed_scale"])
    (DATADIR / "physics_check.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\nmean max|div u|: prediction {summary['mean_pred_max_abs_div']:.3e}"
              f"  vs CFD reference {summary['mean_ref_max_abs_div']:.1e}")
        print(f"mean max no-slip wall velocity leak: {summary['mean_wall_max_abs_u']:.4f}")
        print(f"mean max lid reproduction error:     {summary['mean_lid_max_abs_err']:.4f}")
    return summary


if __name__ == "__main__":
    main()
