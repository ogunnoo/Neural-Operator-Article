"""Evaluate the trained DeepONet on the held-out lid functions.

All metrics are computed against the finite-difference CFD solution, which is
itself an approximation (validated against Ghia et al. -- see
`src/validate_solver.py`).  "Error" below therefore means "disagreement with a
verified but approximate reference", not "distance from the exact
Navier-Stokes solution".
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from .config import CFG, DATADIR, MODELDIR
from .dataset import function_distance, generate
from .deeponet import load_checkpoint, normalise_g


def load_model():
    ckpt = MODELDIR / "deeponet_cavity.msgpack"
    model, params, norm, cfg_dict = load_checkpoint(ckpt, CFG.n_sensors)

    @jax.jit
    def fields(g, coords):
        """g: (N, m) input functions, coords: (M, 2) -> (N, M, 2) velocities."""
        uvn = model.apply({"params": params}, normalise_g(g, norm), coords)
        return uvn * norm["uv_std"] + norm["uv_mean"]

    return fields, cfg_dict


def predict_fields(fields_fn, g, x, y):
    """Return (u, v) on the (len(y), len(x)) grid for one lid profile."""
    X, Y = np.meshgrid(x, y)
    coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1), jnp.float32)
    gb = jnp.asarray(np.atleast_2d(g), jnp.float32)          # (1, m)
    uv = np.asarray(fields_fn(gb, coords))[0]                # (M, 2)
    return uv[:, 0].reshape(X.shape), uv[:, 1].reshape(X.shape)


def metrics(u_true, v_true, u_pred, v_pred, x, y):
    """Relative L2 (combined and per-component) plus supporting statistics."""
    def rel_l2(a_true, a_pred):
        return float(np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true))

    num = np.sqrt(((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2).sum())
    den = np.sqrt((u_true**2 + v_true**2).sum())

    speed_t = np.sqrt(u_true**2 + v_true**2)
    speed_p = np.sqrt(u_pred**2 + v_pred**2)

    mid = len(x) // 2
    assert abs(x[mid] - 0.5) < 1e-12
    cl_t, cl_p = u_true[:, mid], u_pred[:, mid]
    cl_rel = float(np.linalg.norm(cl_p - cl_t) / np.linalg.norm(cl_t))

    return {
        "rel_l2_velocity": float(num / den),
        "rel_l2_u": rel_l2(u_true, u_pred),
        "rel_l2_v": rel_l2(v_true, v_pred),
        "mae_u": float(np.abs(u_pred - u_true).mean()),
        "mae_v": float(np.abs(v_pred - v_true).mean()),
        "max_abs_err_u": float(np.abs(u_pred - u_true).max()),
        "max_abs_err_v": float(np.abs(v_pred - v_true).max()),
        "max_abs_speed_err": float(np.abs(speed_p - speed_t).max()),
        "centerline_rel_l2_u": cl_rel,
        "peak_speed_true": float(speed_t.max()),
        "peak_speed_pred": float(speed_p.max()),
    }


def main(verbose: bool = True):
    data = generate(verbose=False)
    x, y = data["x"], data["y"]
    fields_fn, model_cfg = load_model()

    rows = []
    for i in range(len(data["test_g"])):
        g = data["test_g"][i]
        ut, vt = data["test_u"][i], data["test_v"][i]
        up, vp = predict_fields(fields_fn, g, x, y)
        m = metrics(ut, vt, up, vp, x, y)
        m["test_index"] = i + 1
        m["coeffs"] = [round(float(c), 4) for c in data["test_coeffs"][i]]
        m["lid_peak"] = float(g.max())
        rows.append(m)
        if verbose:
            print(f"Test {i+1}: relL2(vel) {m['rel_l2_velocity']:.3%}  "
                  f"u {m['rel_l2_u']:.3%}  v {m['rel_l2_v']:.3%}  "
                  f"maxAbsErr(u) {m['max_abs_err_u']:.4f}  "
                  f"centerline {m['centerline_rel_l2_u']:.3%}")

    # Same metrics on the training-pool functions, for a train/test comparison.
    train_rel = []
    for i in range(len(data["train_g"])):
        up, vp = predict_fields(fields_fn, data["train_g"][i], x, y)
        m = metrics(data["train_u"][i], data["train_v"][i], up, vp, x, y)
        train_rel.append(m["rel_l2_velocity"])

    # How far is each unseen lid from the closest lid the network actually fit?
    # This is what separates "validation" from "held-out test" here: the test
    # profiles were drawn subject to a minimum-separation rule, the validation
    # profiles were not.
    n_fit = len(data["train_g"]) - CFG.n_val
    fit_g = data["train_g"][:n_fit]

    def nearest(g):
        return float(min(function_distance(g, f) for f in fit_g))

    val_sep = [nearest(g) for g in data["train_g"][n_fit:]]
    test_sep = [nearest(g) for g in data["test_g"]]
    for r, sp in zip(rows, test_sep):
        r["dist_to_nearest_fit_lid"] = sp

    summary = {
        "per_test_function": rows,
        "test_mean_rel_l2_velocity": float(np.mean([r["rel_l2_velocity"] for r in rows])),
        "test_median_rel_l2_velocity": float(np.median([r["rel_l2_velocity"] for r in rows])),
        "test_worst_rel_l2_velocity": float(np.max([r["rel_l2_velocity"] for r in rows])),
        "test_best_rel_l2_velocity": float(np.min([r["rel_l2_velocity"] for r in rows])),
        "test_mean_rel_l2_u": float(np.mean([r["rel_l2_u"] for r in rows])),
        "test_mean_rel_l2_v": float(np.mean([r["rel_l2_v"] for r in rows])),
        "n_fit_functions": int(len(train_rel) - CFG.n_val),
        "n_val_functions": int(CFG.n_val),
        "train_pool_mean_rel_l2_velocity": float(np.mean(train_rel[:-CFG.n_val])),
        "val_mean_rel_l2_velocity": float(np.mean(train_rel[-CFG.n_val:])),
        "model_config": model_cfg,
        "val_per_function_rel_l2": train_rel[-CFG.n_val:],
        "val_dist_to_nearest_fit_lid": val_sep,
        "test_dist_to_nearest_fit_lid": test_sep,
        "val_mean_separation": float(np.mean(val_sep)),
        "test_mean_separation": float(np.mean(test_sep)),
        "distance_definition": "2*||g-f||_2/(||g||_2+||f||_2)",
    }
    (DATADIR / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"\nMean test relL2 (velocity): {summary['test_mean_rel_l2_velocity']:.3%}")
        print(f"Train-pool mean relL2:      {summary['train_pool_mean_rel_l2_velocity']:.3%}")
        print(f"Validation mean relL2:      {summary['val_mean_rel_l2_velocity']:.3%}")
    return summary


if __name__ == "__main__":
    main()
