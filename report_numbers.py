#!/usr/bin/env python3
"""Dump every number the article cites, straight from the executed run."""
import json
import numpy as np
from scipy.stats import t
from src.config import CFG, DATADIR, FIGDIR
from src.dataset import function_distance, generate

d = generate(verbose=False)
hist = json.loads((DATADIR / "training_history.json").read_text())
met = json.loads((DATADIR / "test_metrics.json").read_text())
val = json.loads((DATADIR / "solver_validation.json").read_text())

out = {}
out["solver"] = val
all_g = np.concatenate([d["train_g"], d["test_g"]])
mean_lid_speed = np.trapezoid(all_g, d["x"], axis=1)
out["dataset"] = dict(
    n_pool=int(len(d["train_g"])), n_fit=hist["n_fit_functions"],
    n_val=hist["n_val_functions"], n_test=int(len(d["test_g"])),
    grid=f"{CFG.n}x{CFG.n}", grid_points=int(CFG.n * CFG.n),
    re=CFG.re, taper_delta=CFG.taper_delta,
    cfd_seconds=round(float(d["cfd_seconds"]), 1),
    total_solutions=int(len(d["train_g"]) + len(d["test_g"])),
    targets=int(hist["n_fit_functions"] * CFG.n * CFG.n * 2),
    max_solver_residual=float(max(d["train_diag"][:, 1].max(), d["test_diag"][:, 1].max())),
    max_divergence=float(max(d["train_diag"][:, 2].max(), d["test_diag"][:, 2].max())),
    mean_solver_steps=float(np.mean(np.concatenate([d["train_diag"][:, 0], d["test_diag"][:, 0]]))),
    lid_peak_min=float(all_g.max(1).min()), lid_peak_max=float(all_g.max(1).max()),
    mean_lid_speed_min=float(mean_lid_speed.min()),
    mean_lid_speed_max=float(mean_lid_speed.max()),
    peak_speed_re_min=float(CFG.re * all_g.max(1).min()),
    peak_speed_re_max=float(CFG.re * all_g.max(1).max()),
    mean_speed_re_min=float(CFG.re * mean_lid_speed.min()),
    mean_speed_re_max=float(CFG.re * mean_lid_speed.max()),
    psi_min_range=[float(d["train_psi"].reshape(len(d["train_g"]), -1).min(1).min()),
                   float(d["train_psi"].reshape(len(d["train_g"]), -1).min(1).max())],
)
out["training"] = dict(
    n_params=hist["n_params"], steps=hist["steps"],
    points_per_step=hist["points_per_step"], lr=hist["lr"],
    weight_decay=hist["weight_decay"], grad_clip=hist["grad_clip"],
    seconds=round(hist["train_seconds"], 1),
    minutes=round(hist["train_seconds"] / 60, 1),
    best_val=hist["best_val_rel_l2"], best_step=hist["best_step"],
    backend=hist["backend"], seed=hist["seed"],
    final_train_mse=hist["train_mse"][-1],
    model_config=met["model_config"],
)
out["metrics"] = met
test_errs = np.array([r["rel_l2_velocity"] for r in met["per_test_function"]])
test_se = float(test_errs.std(ddof=1) / np.sqrt(len(test_errs)))
out["test_uncertainty"] = dict(
    median=float(np.median(test_errs)),
    sample_sd=float(test_errs.std(ddof=1)),
    standard_error=test_se,
    mean_t95_ci=[float(x) for x in t.interval(
        0.95, len(test_errs) - 1, loc=float(test_errs.mean()), scale=test_se
    )],
    note="descriptive t interval over six deliberately separated challenge cases",
)

# separation of test functions from the training pool
sep = [min(function_distance(g, t) for t in d["train_g"][:met["n_fit_functions"]])
       for g in d["test_g"]]
out["test_separation_rel_l2"] = [round(s, 4) for s in sep]
out["test_separation_min"] = round(min(sep), 4)
out["distance_definition"] = "2*||g-f||_2/(||g||_2+||f||_2)"

# Inference timing: one forward pass vs one CFD solve.
# Wall-clock timings are machine- and load-dependent, so take medians over
# repeats rather than a single sample.
import time, jax, jax.numpy as jnp
from src.evaluate import load_model, predict_fields
from src.cavity_solver import solve_cavity

fields_fn, _ = load_model()
x, y = d["x"], d["y"]
_ = predict_fields(fields_fn, d["test_g"][0], x, y)  # trigger JIT compilation

inf = []
for _ in range(10):
    t0 = time.time()
    for _ in range(50):
        predict_fields(fields_fn, d["test_g"][0], x, y)
    inf.append((time.time() - t0) / 50 * 1000)
out["inference_ms_per_field"] = round(float(np.median(inf)), 2)

cfd = []
for k in range(5):
    t0 = time.time()
    solve_cavity(d["test_g"][k % len(d["test_g"])], re=CFG.re, n=CFG.n)
    cfd.append((time.time() - t0) * 1000)
out["cfd_ms_per_field"] = round(float(np.median(cfd)), 1)
out["speedup"] = round(out["cfd_ms_per_field"] / out["inference_ms_per_field"], 1)
out["timing_note"] = ("medians over repeats; wall-clock on one machine, "
                      "varies with load and hardware")
# How many new solves before training pays for itself?
_overhead = out["training"]["seconds"] + out["dataset"]["cfd_seconds"]
_saving = (out["cfd_ms_per_field"] - out["inference_ms_per_field"]) / 1000.0
out["breakeven_solves"] = int(round(_overhead / _saving))

# The marginal calculation above deliberately answers the deployment question:
# how many new cases amortise the selected final run? For transparent accounting,
# also report every recorded model-selection run. The earlier failed prototype is
# not timed in machine-readable form and is therefore excluded from this lower bound.
sweep_seconds = float(sum(r["seconds"] for r in json.loads(
    (DATADIR / "sweep_results.json").read_text())))
candidate_seconds = float(sum(json.loads(
    (DATADIR / f"candidate_{tag}.json").read_text())["seconds"]
    for tag in ("A", "B")))
recorded_development_seconds = sweep_seconds + candidate_seconds + out["dataset"]["cfd_seconds"]
out["recorded_model_selection_seconds"] = round(recorded_development_seconds, 1)
out["recorded_model_selection_breakeven_solves"] = int(round(
    recorded_development_seconds / _saving))
out["recorded_model_selection_note"] = (
    "lower bound: sweep + candidates A/B + CFD dataset; excludes the untimed failed prototype"
)

out["figures"] = sorted(p.name for p in FIGDIR.glob("*.png"))
(DATADIR / "article_numbers.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
