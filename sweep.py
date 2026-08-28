#!/usr/bin/env python3
"""Small hyperparameter sweep. Selection is on VALIDATION functions only;
the 6 held-out test functions are never touched here."""
import itertools, json, time
from src.config import CFG, DATADIR
from src import train

GRID = [
    # (steps, points/step, lr, weight_decay, latent, b_hid, b_dep, t_hid, t_dep)
    (30_000,  512, 1e-3, 1e-4, 128, 128, 3, 160, 4),
    (30_000, 1681, 1e-3, 1e-4, 128, 128, 3, 160, 4),
    (60_000,  512, 1e-3, 1e-4, 128, 128, 3, 160, 4),
    (60_000, 1681, 1e-3, 1e-4, 128, 128, 3, 160, 4),
    (60_000, 1681, 2e-3, 1e-4, 128, 128, 3, 160, 4),
    (60_000, 1681, 1e-3, 1e-3, 128, 128, 3, 160, 4),
    (60_000, 1681, 1e-3, 1e-4, 192, 128, 3, 200, 5),
    (60_000, 1681, 1e-3, 1e-4,  64, 128, 3, 160, 4),
    (60_000, 1681, 1e-3, 1e-4, 128, 192, 4, 200, 5),
    (120_000, 1681, 1e-3, 1e-4, 128, 128, 3, 200, 5),
]

results = []
for i, (st, pps, lr, wd, lat, bh, bd, th, td) in enumerate(GRID, 1):
    CFG.steps, CFG.points_per_step, CFG.lr, CFG.weight_decay = st, pps, lr, wd
    CFG.latent, CFG.branch_hidden, CFG.branch_depth = lat, bh, bd
    CFG.trunk_hidden, CFG.trunk_depth = th, td
    t0 = time.time()
    h = train.main(verbose=False, save=False)
    r = dict(cfg=dict(steps=st, points_per_step=pps, lr=lr, weight_decay=wd,
                      latent=lat, branch_hidden=bh, branch_depth=bd,
                      trunk_hidden=th, trunk_depth=td),
             best_val=h["best_val_rel_l2"], best_step=h["best_step"],
             final_val=h["val_rel_l2"][-1], final_mse=h["train_mse"][-1],
             n_params=h["n_params"], seconds=round(time.time()-t0, 1))
    results.append(r)
    print(f"[{i}/{len(GRID)}] best_val {r['best_val']:.4%} (step {r['best_step']}) "
          f"final_val {r['final_val']:.4%} params {r['n_params']:,} "
          f"{r['seconds']}s :: {r['cfg']}", flush=True)

results.sort(key=lambda r: r["best_val"])
(DATADIR / "sweep_results.json").write_text(json.dumps(results, indent=2))
print("\nBEST:", json.dumps(results[0], indent=2))
