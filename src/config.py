"""Central configuration for the lid-driven-cavity neural-operator experiment.

Everything that controls the experiment lives here so that a run is fully
described by a single object and is reproducible from a fresh checkout.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

# Project root = parent of src/
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIGDIR = OUT / "figures"
DATADIR = OUT / "data"
MODELDIR = OUT / "models"

for _d in (FIGDIR, DATADIR, MODELDIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # ---------------- Physics / discretisation ----------------
    # The nondimensional PDE coefficient is fixed for the whole study. Re is
    # defined with U_ref = L = 1, so nu = 1/Re is identical for every sample.
    # Because the imposed lid amplitude varies, case-specific Reynolds scales
    # based on actual peak or mean lid speed vary; report_numbers.py records both.
    re: float = 100.0
    n: int = 41                     # grid is n x n over the unit square
    cfl: float = 0.35               # safety factor on the explicit time step
    steady_tol: float = 1e-6        # ||dw/dt||_inf threshold for steady state
    max_steps: int = 200_000

    # ---------------- Input-function family ----------------
    # g(x) = a0 + a1 sin(pi x) + a2 sin(2 pi x) + a3 cos(pi x), corner-tapered.
    a0_range: tuple = (0.80, 1.20)
    a_amp: float = 0.35             # per-coefficient bound for a1, a2, a3
    shape_budget: float = 0.60      # |a1|+|a2|+|a3| <= shape_budget * a0
    taper_delta: float = 0.05       # corner taper width (in units of x)

    n_train: int = 64          # pool; split into fit + validation
    n_val: int = 8             # monitoring/model selection, never the test set
    n_test: int = 6            # held out until final evaluation

    # ---------------- DeepONet ----------------
    n_sensors: int = 41             # = n, lid sampled on the grid x-nodes
    latent: int = 128               # p, the number of basis functions per field
    branch_hidden: int = 128
    branch_depth: int = 3
    trunk_hidden: int = 160
    trunk_depth: int = 4

    # ---------------- Training ----------------
    # One "step" uses ALL fitting functions and a random subset of grid points,
    # which is the natural minibatch for the outer-product DeepONet form.
    steps: int = 120_000
    points_per_step: int = 1681   # = n*n, i.e. all grid points each step
    lr: float = 2e-3
    lr_final: float = 1e-5          # cosine-annealed target
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    seed: int = 7
    device: str = "auto"            # "auto" | "cpu" | "mps" | "cuda"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


CFG = Config()
