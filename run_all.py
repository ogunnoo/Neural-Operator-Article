#!/usr/bin/env python3
"""Reproduce the entire experiment end to end.

    python run_all.py            # uses cached CFD data / model if present
    python run_all.py --fresh    # regenerates everything from scratch

Stages: solver validation -> CFD dataset -> DeepONet training -> evaluation
-> figures.
"""

import argparse
import json
import sys

from src.config import CFG, DATADIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="ignore cached dataset and model")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    if not args.skip_validation:
        print("=" * 70)
        print("STAGE 1  solver validation against Ghia et al. (1982), Re=100")
        print("=" * 70)
        from src.validate_solver import run as validate
        rep = validate(n=CFG.n)
        print(json.dumps(rep, indent=2))
        (DATADIR / "solver_validation.json").write_text(json.dumps(rep, indent=2))
        if rep["u_rel_l2_vs_ghia"] > 0.05:
            print("WARNING: solver deviates >5% from the benchmark", file=sys.stderr)

    print("\n" + "=" * 70)
    print("STAGE 2  CFD dataset generation")
    print("=" * 70)
    from src.dataset import generate, dataset_path
    generate(force=args.fresh)

    print("\n" + "=" * 70)
    print("STAGE 3  DeepONet training")
    print("=" * 70)
    from src.config import MODELDIR
    ckpt = MODELDIR / "deeponet_cavity.msgpack"
    if ckpt.exists() and not args.fresh:
        print(f"using existing checkpoint {ckpt.name} (pass --fresh to retrain)")
    else:
        from src.train import main as train_main
        train_main()

    print("\n" + "=" * 70)
    print("STAGE 4  evaluation on held-out lid functions")
    print("=" * 70)
    from src.evaluate import main as eval_main
    eval_main()

    print("\n" + "=" * 70)
    print("STAGE 5  physics check (unconstrained model vs incompressibility)")
    print("=" * 70)
    from src.physics_check import main as phys_main
    phys_main()

    print("\n" + "=" * 70)
    print("STAGE 6  figures")
    print("=" * 70)
    from src.plots import main as plots_main
    plots_main()

    print("\n" + "=" * 70)
    print("STAGE 7  article numbers")
    print("=" * 70)
    import subprocess, sys as _s
    subprocess.run([_s.executable, "report_numbers.py"], check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
