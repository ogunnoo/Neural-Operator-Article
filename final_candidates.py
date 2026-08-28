#!/usr/bin/env python3
"""Run the two best sweep configurations for longer. Selection on validation only."""
import json, sys, time
from src.config import CFG, DATADIR
from src import train

CANDS = {
    "_A": dict(steps=120_000, points_per_step=1681, lr=2e-3, weight_decay=1e-4,
               latent=128, branch_hidden=128, branch_depth=3,
               trunk_hidden=200, trunk_depth=5),
    "_B": dict(steps=120_000, points_per_step=1681, lr=2e-3, weight_decay=1e-4,
               latent=128, branch_hidden=128, branch_depth=3,
               trunk_hidden=160, trunk_depth=4),
}
tag = sys.argv[1]
for k, v in CANDS[tag].items():
    setattr(CFG, k, v)
t0 = time.time()
h = train.main(verbose=True, save=True, tag=tag)
print(f"[{tag}] best_val {h['best_val_rel_l2']:.4%} at step {h['best_step']} "
      f"in {time.time()-t0:.0f}s")
(DATADIR / f"candidate{tag}.json").write_text(json.dumps(
    {"tag": tag, "cfg": CANDS[tag], "best_val": h["best_val_rel_l2"],
     "best_step": h["best_step"], "seconds": h["train_seconds"],
     "n_params": h["n_params"]}, indent=2))
