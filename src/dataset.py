"""The family of lid functions g(x), and the cached CFD dataset built from it.

The object that varies across the dataset is an entire *function* -- the
horizontal velocity profile along the moving top lid -- not a scalar. Reynolds
number is held fixed at Re = 100 throughout, so nothing about the PDE changes
between samples except the top boundary condition.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .cavity_solver import solve_cavity, grid
from .config import CFG, DATADIR


# --------------------------------------------------------------------------
# Input-function family
# --------------------------------------------------------------------------

def corner_taper(x: np.ndarray, delta: float) -> np.ndarray:
    """Smooth window that vanishes at x = 0 and x = 1.

    The classical lid-driven cavity has a genuine discontinuity at the two top
    corners: the lid moves at u = 1 while the side walls are no-slip.  That
    singularity is not resolvable on any finite grid -- it produces a wall
    vorticity that grows without bound as h -> 0 and contaminates the reference
    solution near the corners.

    We therefore use the standard "regularised cavity" trick and multiply every
    profile by tanh(x/delta) * tanh((1-x)/delta).  With delta = 0.05 the window
    is above 0.99 for x in [0.15, 0.85], so the shape of g is essentially
    untouched over the middle 70% of the lid; only a thin corner layer, a few
    grid cells wide, is smoothed.  Every profile in the family gets the same
    window, so the taper is part of the operator's input distribution rather
    than a per-sample fudge.
    """
    return np.tanh(x / delta) * np.tanh((1.0 - x) / delta)


def lid_profile(x: np.ndarray, coeffs, delta: float = None) -> np.ndarray:
    """g(x) = [a0 + a1 sin(pi x) + a2 sin(2 pi x) + a3 cos(pi x)] * taper(x)."""
    if delta is None:
        delta = CFG.taper_delta
    a0, a1, a2, a3 = coeffs
    raw = (
        a0
        + a1 * np.sin(np.pi * x)
        + a2 * np.sin(2.0 * np.pi * x)
        + a3 * np.cos(np.pi * x)
    )
    g = raw * corner_taper(x, delta)
    g[0] = 0.0
    g[-1] = 0.0
    return g


def sample_coeffs(rng: np.random.Generator):
    """Draw one coefficient vector from the family.

    The `shape_budget` rejection rule keeps |a1|+|a2|+|a3| <= 0.6 * a0, which
    guarantees the untapered profile stays positive (min >= 0.4 * a0 >= 0.32).
    The lid therefore always moves in one direction and the flow stays a single
    primary vortex plus corner eddies. The PDE coefficient Re=100 is fixed, but
    Reynolds scales formed from each case's actual lid speed are not constant.
    """
    while True:
        a0 = rng.uniform(*CFG.a0_range)
        a1, a2, a3 = rng.uniform(-CFG.a_amp, CFG.a_amp, size=3)
        if abs(a1) + abs(a2) + abs(a3) <= CFG.shape_budget * a0:
            return (float(a0), float(a1), float(a2), float(a3))


def function_distance(g1: np.ndarray, g2: np.ndarray) -> float:
    """Symmetric normalised L2 distance between two sampled lid functions."""
    return float(
        2.0 * np.linalg.norm(g1 - g2)
        / (np.linalg.norm(g1) + np.linalg.norm(g2))
    )


def build_function_family(rng: np.random.Generator, x: np.ndarray,
                          n_train: int, n_test: int, min_sep: float = 0.12):
    """Draw training profiles, then held-out test profiles that are provably
    distinct from every training profile.

    `min_sep` is the symmetric normalised L2 distance
    2||g1-g2||/(||g1||+||g2||). A candidate test profile is rejected unless it
    is at least 12% away from every
    training profile *and* from every already-accepted test profile.  This is
    what makes "unseen" mean something: the test lids remain within the same
    four-coefficient support, but separation conditioning makes them a challenge
    set rather than an IID sample from the base coefficient distribution.
    """
    train = [sample_coeffs(rng) for _ in range(n_train)]
    train_g = [lid_profile(x, c) for c in train]

    test, test_g = [], []
    attempts = 0
    while len(test) < n_test:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not find sufficiently separated test profiles")
        c = sample_coeffs(rng)
        g = lid_profile(x, c)
        if min(function_distance(g, t) for t in train_g) < min_sep:
            continue
        if test_g and min(function_distance(g, t) for t in test_g) < min_sep:
            continue
        test.append(c)
        test_g.append(g)

    return train, np.array(train_g), test, np.array(test_g)


# --------------------------------------------------------------------------
# Dataset generation with caching
# --------------------------------------------------------------------------

def _cache_key() -> str:
    """Hash the settings that actually change the CFD data."""
    rel = dict(
        re=CFG.re, n=CFG.n, cfl=CFG.cfl, tol=CFG.steady_tol,
        a0=CFG.a0_range, amp=CFG.a_amp, budget=CFG.shape_budget,
        delta=CFG.taper_delta, n_train=CFG.n_train, n_test=CFG.n_test,
        seed=CFG.seed, v=3,
    )
    return hashlib.sha1(json.dumps(rel, sort_keys=True).encode()).hexdigest()[:10]


def dataset_path() -> Path:
    return DATADIR / f"cavity_dataset_{_cache_key()}.npz"


def generate(force: bool = False, verbose: bool = True) -> dict:
    """Generate (or load from cache) the full operator-learning dataset."""
    path = dataset_path()
    if path.exists() and not force:
        if verbose:
            print(f"[dataset] loading cached CFD data: {path.name}")
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    rng = np.random.default_rng(CFG.seed)
    x, y = grid(CFG.n)

    train_c, train_g, test_c, test_g = build_function_family(
        rng, x, CFG.n_train, CFG.n_test
    )

    def run_split(profiles, name):
        us, vs, psis, diag = [], [], [], []
        for i, g in enumerate(profiles):
            t0 = time.time()
            sol = solve_cavity(
                g, re=CFG.re, n=CFG.n, cfl=CFG.cfl,
                tol=CFG.steady_tol, max_steps=CFG.max_steps,
            )
            if not sol.converged:
                raise RuntimeError(
                    f"{name} case {i} did not reach steady state "
                    f"(residual={sol.residual:.3e})"
                )
            us.append(sol.u)
            vs.append(sol.v)
            psis.append(sol.psi)
            diag.append((sol.steps, sol.residual, sol.max_divergence))
            if verbose:
                print(
                    f"[dataset] {name} {i+1:>3}/{len(profiles)}  "
                    f"steps={sol.steps:>6}  res={sol.residual:.2e}  "
                    f"|div|={sol.max_divergence:.1e}  {time.time()-t0:.2f}s"
                )
        return (np.array(us, dtype=np.float64),
                np.array(vs, dtype=np.float64),
                np.array(psis, dtype=np.float64),
                np.array(diag, dtype=np.float64))

    if verbose:
        print(f"[dataset] generating {CFG.n_train} train + {CFG.n_test} test "
              f"CFD solutions on a {CFG.n}x{CFG.n} grid at Re={CFG.re:g}")

    t_start = time.time()
    tr_u, tr_v, tr_psi, tr_diag = run_split(train_g, "train")
    te_u, te_v, te_psi, te_diag = run_split(test_g, "test ")
    cfd_seconds = time.time() - t_start

    out = dict(
        x=x, y=y,
        train_g=train_g, train_u=tr_u, train_v=tr_v, train_psi=tr_psi,
        train_coeffs=np.array(train_c), train_diag=tr_diag,
        test_g=test_g, test_u=te_u, test_v=te_v, test_psi=te_psi,
        test_coeffs=np.array(test_c), test_diag=te_diag,
        cfd_seconds=np.array(cfd_seconds),
    )
    np.savez_compressed(path, **out)
    if verbose:
        print(f"[dataset] wrote {path} ({path.stat().st_size/1e6:.1f} MB) "
              f"in {cfd_seconds:.1f}s of CFD")
    return out


if __name__ == "__main__":
    d = generate()
    print("train_u", d["train_u"].shape, "test_u", d["test_u"].shape)
