# Learning the Lid-Driven Cavity Solution Operator with a DeepONet

A small, self-contained, reproducible experiment that learns the **function-to-function** map

```
g(x)  --->  [ u(x,y), v(x,y) ]
```

where `g(x)` is the horizontal velocity profile along the moving top lid of a 2D
lid-driven cavity and `(u, v)` is the resulting steady interior velocity field.
The nondimensional PDE coefficient is fixed at **Re = 100** (`nu = 0.01`),
while the boundary function varies within a four-coefficient smooth family.
Because lid amplitude varies, Reynolds scales based on each case's actual lid
speed are not constant.

This is the companion experiment to an article contrasting **PINNs** (which
learn one solution to one PDE instance) with **neural operators** (which learn a
mapping between problems and their solutions).

The article generated from this code is [`neural_operator_experiment.md`](neural_operator_experiment.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the whole pipeline and reuses cached artefacts where possible:

| Stage | What it does | Cost |
|---|---|---|
| 1 | Validate the CFD solver against Ghia et al. (1982) | ~4 s |
| 2 | Generate 70 steady cavity solutions (cached to `outputs/data/`) | ~32 s |
| 3 | Train the DeepONet | ~4 min (CPU) |
| 4 | Evaluate on 6 held-out lid functions | ~1 s |
| 5 | Physics check (divergence, wall/lid conditions) | ~1 s |
| 6 | Render all figures | ~30 s |
| 7 | Dump the article's numbers to JSON | ~15 s |

Use `python run_all.py --fresh` to ignore the caches and rebuild everything.

## Layout

```
neural_operator_cavity/
├── README.md
├── requirements.txt
├── run_all.py                  # end-to-end pipeline
├── build_pdf.sh                # article markdown -> print-ready PDF
├── sweep.py                    # hyperparameter search (validation-only selection)
├── final_candidates.py         # longer runs of the two best sweep configs
├── report_numbers.py           # dumps every number the article cites
├── src/
│   ├── config.py               # every knob for the experiment
│   ├── cavity_solver.py        # steady Navier-Stokes, streamfunction-vorticity
│   ├── validate_solver.py      # benchmark check vs Ghia et al. (1982)
│   ├── dataset.py              # the family of lid functions g(x) + cached CFD data
│   ├── deeponet.py             # branch/trunk model (JAX + Flax)
│   ├── train.py                # supervised operator training (Adam/optax)
│   ├── evaluate.py             # relative L2 and other metrics
│   ├── physics_check.py        # does the unconstrained model respect the physics?
│   └── plots.py                # every figure in the article
└── outputs/
    ├── figures/                # PNGs used by the article
    ├── data/                   # cached CFD dataset, metrics, training history
    └── models/                 # trained DeepONet checkpoint
```

## The numerical reference solver

`src/cavity_solver.py` uses the **streamfunction-vorticity** formulation rather
than a primitive-variable projection method. Because `u = ∂ψ/∂y` and
`v = -∂ψ/∂x`, the discrete velocity field is divergence-free *by construction* —
measured `|∇·u|` is ~1e-15 (machine precision) rather than "small if the
pressure solve converged". Since these fields are the ground truth the operator
is asked to reproduce, that structural guarantee matters.

The solver marches vorticity transport in pseudo-time until `‖∂ω/∂t‖∞ < 1e-6`,
so every stored sample is genuinely steady rather than merely "run for N steps".
It is validated against the standard Ghia, Ghia & Shin (1982) Re=100 benchmark
in `src/validate_solver.py`.

It is still an educational solver, not production CFD: second-order central
differences on a uniform 41×41 grid, first-order (Thom) wall vorticity. See the
article's limitations section.

## Framework note

The operator network is written in **JAX/Flax** with **optax** for optimisation.
The CFD solver stays in NumPy/SciPy — it leans on `scipy.sparse.linalg.splu` to
prefactorise the streamfunction Poisson operator once and reuse the LU factors
every pseudo-time step, which has no JAX equivalent and is not the part of the
pipeline that benefits from autodiff or accelerators.

The DeepONet is written in the *outer-product* form: the branch runs once per
input function and the trunk once per query point, and they meet in an
`einsum`. This is both the clearest statement of the DeepONet ansatz and about
two orders of magnitude cheaper than evaluating the branch once per
(function, point) pair.

## Reproducibility

Every random choice is seeded from `Config.seed` (default 7): the lid
coefficients, the train/validation/test split, network initialisation, and
minibatch sampling. The CFD dataset is cached under a hash of the settings that
affect it, so changing a network hyperparameter does not trigger a CFD rerun.

Model selection uses a validation split of the training pool. The 6 held-out
test functions are not evaluated until `src/evaluate.py` runs.

## Headline results

| | |
|---|---|
| Solver vs Ghia et al. (Re=100, 41×41) | 0.56% (u), 1.62% (v) relative L2 |
| Training data | 56 lid functions, fixed `nu=0.01` |
| Model | 190,402 parameters (branch 128×3, trunk 160×4, p=128) |
| Training | 120k steps, ~35 min on CPU |
| Mean error on 6 separated challenge lids | **1.28%** relative L2 velocity (median 1.59%) |
| Inference vs CFD solve | ~1.9 ms vs ~498 ms (~260×) |

One result worth highlighting, because it is the point of the article's
"Where did the physics go?" section: the trained model reproduces supervised
boundary values accurately (mean maximum stationary-wall speed ~0.0009), but
its outputs are **not** divergence-free — mean max |∇·u| ≈ 4.3e-2 and mean RMS
divergence ≈ 7.7e-3, against ~3e-15 for the reference fields. Accurate in norm,
wrong in structure. Run
`python -m src.physics_check` to reproduce.

## Building the PDF

`neural_operator_experiment.md` renders to a print-ready PDF with:

```bash
./build_pdf.sh
```

The pipeline is pandoc → self-contained HTML (KaTeX math, base64-inlined
figures) → headless Chrome → `reportlab` footer stamping. It needs
`brew install pandoc` and Google Chrome, but **no LaTeX distribution**. Network
access is required only on the first run, so pandoc can inline the KaTeX
assets. Layout is controlled by `build/print.css`.

Output: `Neural Operator Experiment - Lid-Driven Cavity DeepONet.pdf` (16 pages, A4).

## Provenance

- `outputs/data/sweep_results.json` — the 10-configuration search, scored on
  validation functions only.
- `outputs/data/candidate_A.json`, `candidate_B.json` — the two long runs; B
  won (0.137% vs 0.143% validation) and is the shipped checkpoint.
- `outputs/data/first_run_log_superseded.txt` — the first, badly-tuned training
  run (loss spike, 5.4% validation error), kept because the article refers to
  it.
