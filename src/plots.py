"""All figures for the article. Every plot is produced from executed code."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .config import CFG, FIGDIR, DATADIR
from .dataset import generate
from .evaluate import load_model, predict_fields, metrics

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})

SPEED_CMAP = "viridis"
ERR_CMAP = "magma"


def _speed_panel(ax, X, Y, u, v, vmax, title, cmap=SPEED_CMAP):
    speed = np.sqrt(u**2 + v**2)
    im = ax.contourf(X, Y, speed, levels=np.linspace(0, vmax, 41),
                     cmap=cmap, extend="max")
    ax.streamplot(X, Y, u, v, density=0.9, color="white",
                  linewidth=0.5, arrowsize=0.6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return im


def _cbar(fig, im, ax, vmax=None, fmt=None):
    """Colourbar with few ticks, at enough precision to stay distinguishable."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if vmax is not None:
        cb.set_ticks(np.linspace(0, vmax, 5))
    else:
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
    ticks = cb.get_ticks()
    if fmt is None:
        # Choose decimals so adjacent ticks never render as the same string.
        span = float(np.max(ticks) - np.min(ticks)) or 1.0
        dec = max(2, int(np.ceil(-np.log10(span / len(ticks)))) + 1)
        fmt = f"%.{min(dec, 6)}f"
    cb.ax.set_yticklabels([fmt % t for t in ticks])
    cb.ax.tick_params(labelsize=7)
    return cb


# --------------------------------------------------------------------------
# Figure 1 - the input function family
# --------------------------------------------------------------------------

def fig_input_functions(data):
    x = data["x"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    for i, g in enumerate(data["train_g"]):
        ax.plot(x, g, color="0.72", lw=0.8, zorder=1,
                label=f"training pool ({len(data['train_g'])} lid functions)" if i == 0 else None)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, g in enumerate(data["test_g"]):
        ax.plot(x, g, lw=2.2, color=colors[i], zorder=3,
                label=f"held-out test {i+1}")
    ax.set_xlabel("$x$ along the moving lid")
    ax.set_ylabel("lid velocity $g(x)$")
    ax.set_title("The operator's input is an entire function, not a number")
    ax.legend(fontsize=7, ncol=2, loc="lower center", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.axhline(0, color="0.4", lw=0.6)

    # Corner taper detail: why the profiles vanish at x = 0 and x = 1.
    ax = axes[1]
    from .dataset import corner_taper
    xf = np.linspace(0, 1, 801)
    ax.plot(xf, corner_taper(xf, CFG.taper_delta), color="C3", lw=2,
            label=rf"taper $\tanh(x/\delta)\tanh((1-x)/\delta)$, $\delta={CFG.taper_delta}$")
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.fill_between([0, 0.15], 0, 1.05, color="C3", alpha=0.08)
    ax.fill_between([0.85, 1.0], 0, 1.05, color="C3", alpha=0.08)
    ax.set_xlabel("$x$")
    ax.set_ylabel("taper weight")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(0, 1)
    ax.set_title("Corner regularisation\n(shaded: the only region materially altered)")
    ax.legend(fontsize=7, loc="lower center")

    p = FIGDIR / "fig1_input_functions.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 2 - training loss
# --------------------------------------------------------------------------

def fig_training_loss():
    hist = json.loads((DATADIR / "training_history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    ax.semilogy(hist["train_steps"], hist["train_mse"], color="C0", lw=1.2)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("MSE (normalised units)")
    ax.set_title("DeepONet training loss")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    ax.semilogy(hist["val_steps"], np.array(hist["val_rel_l2"]) * 100,
                color="C1", lw=1.4,
                label=f"validation ({hist['n_val_functions']} lids held out of the fit)")
    ax.axvline(hist["best_step"], color="0.5", ls="--", lw=1,
               label=f"selected step {hist['best_step']:,}")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("relative $L_2$ velocity error (%)")
    ax.set_title("Validation error during training")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)

    p = FIGDIR / "fig2_training_loss.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figures 3+ - per-test-function multi-panel summaries
# --------------------------------------------------------------------------

def fig_test_summary(data, fields_fn, i):
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y)
    g = data["test_g"][i]
    ut, vt = data["test_u"][i], data["test_v"][i]
    up, vp = predict_fields(fields_fn, g, x, y)
    m = metrics(ut, vt, up, vp, x, y)

    speed_t = np.sqrt(ut**2 + vt**2)
    speed_p = np.sqrt(up**2 + vp**2)
    vmax = float(max(speed_t.max(), speed_p.max()))
    err = np.sqrt((up - ut) ** 2 + (vp - vt) ** 2)

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.6))
    fig.subplots_adjust(wspace=0.55)

    # (a) the input function
    ax = axes[0]
    for gt in data["train_g"]:
        ax.plot(x, gt, color="0.86", lw=0.6, zorder=1)
    ax.plot(x, g, color="C3", lw=2.4, zorder=3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    ax.set_title(f"(a) input lid function\ntest {i+1} (grey: training pool)")
    ax.set_xlim(0, 1)
    ax.axhline(0, color="0.4", lw=0.6)

    # (b) CFD reference
    ax = axes[1]
    im = _speed_panel(ax, X, Y, ut, vt, vmax, "(b) CFD reference $|\\mathbf{u}|$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    _cbar(fig, im, ax, vmax)

    # (c) DeepONet prediction
    ax = axes[2]
    im = _speed_panel(ax, X, Y, up, vp, vmax, "(c) DeepONet prediction $|\\mathbf{u}|$")
    ax.set_xlabel("$x$")
    _cbar(fig, im, ax, vmax)

    # (d) absolute error
    ax = axes[3]
    im = ax.contourf(X, Y, err, levels=40, cmap=ERR_CMAP)
    ax.set_aspect("equal")
    ax.set_title("(d) $|\\mathbf{u}_{pred}-\\mathbf{u}_{ref}|$")
    ax.set_xlabel("$x$")
    _cbar(fig, im, ax)

    # (e) vertical centreline
    ax = axes[4]
    mid = len(x) // 2
    ax.plot(ut[:, mid], y, color="k", lw=1.8, label="CFD reference")
    ax.plot(up[:, mid], y, color="C3", lw=1.6, ls="--", label="DeepONet")
    ax.set_xlabel("$u(0.5,\\, y)$")
    ax.set_ylabel("$y$")
    ax.set_title("(e) vertical centreline")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Test function {i+1}  —  relative $L_2$ velocity error "
        f"{m['rel_l2_velocity']:.2%}   ($u$: {m['rel_l2_u']:.2%},  "
        f"$v$: {m['rel_l2_v']:.2%})",
        y=1.04, fontsize=12,
    )
    p = FIGDIR / f"fig3_test{i+1}_summary.png"
    fig.savefig(p)
    plt.close(fig)
    return p, m


# --------------------------------------------------------------------------
# Figure 4 - one network, several input functions side by side
# --------------------------------------------------------------------------

def fig_operator_gallery(data, fields_fn, indices=(0, 1, 2, 3)):
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y)
    k = len(indices)

    fig, axes = plt.subplots(3, k, figsize=(3.1 * k, 9.0))
    vmax = 0.0
    preds = []
    for i in indices:
        up, vp = predict_fields(fields_fn, data["test_g"][i], x, y)
        preds.append((up, vp))
        vmax = max(vmax, float(np.sqrt(up**2 + vp**2).max()),
                   float(np.sqrt(data["test_u"][i]**2 + data["test_v"][i]**2).max()))

    for col, i in enumerate(indices):
        g = data["test_g"][i]
        up, vp = preds[col]
        ut, vt = data["test_u"][i], data["test_v"][i]

        ax = axes[0, col]
        ax.plot(x, g, color=f"C{col}", lw=2.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.85)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(f"input  $g_{{{col+1}}}(x)$", fontsize=10)
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("lid velocity $g(x)$")

        ax = axes[1, col]
        im = _speed_panel(ax, X, Y, up, vp, vmax,
                          f"DeepONet  $\\mathcal{{G}}_\\theta[g_{{{col+1}}}]$")
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$y$")

        ax = axes[2, col]
        _speed_panel(ax, X, Y, ut, vt, vmax,
                     f"CFD reference  $\\mathcal{{G}}[g_{{{col+1}}}]$")
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$y$")

    fig.subplots_adjust(right=0.9, hspace=0.32, wspace=0.22)
    cax = fig.add_axes([0.92, 0.12, 0.014, 0.5])
    cb = fig.colorbar(im, cax=cax, label="speed $|\\mathbf{u}|$")
    cb.set_ticks(np.linspace(0, vmax, 6))
    cb.ax.set_yticklabels([f"{t:.2f}" for t in cb.get_ticks()])
    fig.suptitle(
        "One trained network, four unseen lid functions:  "
        "$g_k(x) \;\\longrightarrow\; [u(x,y),\\, v(x,y)]$",
        fontsize=13, y=0.945,
    )
    p = FIGDIR / "fig4_operator_gallery.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 5 - solver validation against Ghia et al.
# --------------------------------------------------------------------------

def fig_solver_validation():
    from .cavity_solver import solve_cavity, grid
    from .validate_solver import GHIA_Y, GHIA_U, GHIA_X, GHIA_V

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    results = {}
    for n, c in zip((33, 41, 65), ("C0", "C3", "C2")):
        x, y = grid(n)
        sol = solve_cavity(np.ones(n), re=CFG.re, n=n)
        mid = n // 2
        results[n] = sol
        axes[0].plot(sol.u[:, mid], y, color=c, lw=1.4,
                     label=f"solver ${n}\\times{n}$")
        axes[1].plot(x, sol.v[mid, :], color=c, lw=1.4,
                     label=f"solver ${n}\\times{n}$")

    axes[0].plot(GHIA_U, GHIA_Y, "ko", ms=4, label="Ghia et al. (1982)")
    axes[0].set_xlabel("$u$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("$u$ on the vertical centreline $x=0.5$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    axes[1].plot(GHIA_X, GHIA_V, "ko", ms=4, label="Ghia et al. (1982)")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$v$")
    axes[1].set_title("$v$ on the horizontal centreline $y=0.5$")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    ax = axes[2]
    n = CFG.n
    x, y = grid(n)
    X, Y = np.meshgrid(x, y)
    sol = results[n]
    im = ax.contourf(X, Y, sol.psi, levels=30, cmap="RdBu_r")
    ax.contour(X, Y, sol.psi, levels=12, colors="k", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"streamfunction, uniform lid, ${n}\\times{n}$\n"
                 f"$\\psi_{{min}}={sol.psi.min():.4f}$ "
                 f"(Ghia: $-0.1034$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.colorbar(im, ax=ax, fraction=0.046)

    p = FIGDIR / "fig5_solver_validation.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Figure 6 - error summary across all test functions
# --------------------------------------------------------------------------

def fig_error_summary():
    s = json.loads((DATADIR / "test_metrics.json").read_text())
    rows = s["per_test_function"]
    idx = [r["test_index"] for r in rows]
    w = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.7))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    ax.bar(np.array(idx) - w, [r["rel_l2_velocity"] * 100 for r in rows],
           w, label="combined $\\mathbf{u}$", color="C0")
    ax.bar(np.array(idx), [r["rel_l2_u"] * 100 for r in rows],
           w, label="$u$", color="C1")
    ax.bar(np.array(idx) + w, [r["rel_l2_v"] * 100 for r in rows],
           w, label="$v$", color="C2")
    ax.set_xticks(idx)
    ax.set_xlabel("held-out test function")
    ax.set_ylabel("relative $L_2$ error (%)")
    ax.set_title("Per-function error on unseen lid profiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    labels = [f"fitting set\n({s['n_fit_functions']})",
              f"validation\n({s['n_val_functions']})",
              f"held-out\ntest ({len(rows)})"]
    vals = [s["train_pool_mean_rel_l2_velocity"] * 100,
            s["val_mean_rel_l2_velocity"] * 100,
            s["test_mean_rel_l2_velocity"] * 100]
    ax.bar(labels, vals, color=["0.6", "C1", "C3"], width=0.55)
    for i, vv in enumerate(vals):
        ax.text(i, vv, f"{vv:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean relative $L_2$ velocity error (%)")
    ax.set_title("Fit vs. generalisation (same trained network)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ax.scatter(np.array(s["val_dist_to_nearest_fit_lid"]) * 100,
               np.array(s["val_per_function_rel_l2"]) * 100,
               s=42, color="C1", label=f"validation ({s['n_val_functions']})",
               zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(np.array(s["test_dist_to_nearest_fit_lid"]) * 100,
               [r["rel_l2_velocity"] * 100 for r in rows],
               s=42, color="C3", marker="s", label=f"held-out test ({len(rows)})",
               zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("distance to nearest fitted lid function\n(relative $L_2$, %)")
    ax.set_ylabel("relative $L_2$ velocity error (%)")
    # Report the actual rank correlation rather than asserting a clean trend:
    # it is strong but has visible exceptions.
    from scipy.stats import spearmanr
    _d = np.array(s["val_dist_to_nearest_fit_lid"] + s["test_dist_to_nearest_fit_lid"])
    _e = np.array(s["val_per_function_rel_l2"]
                  + [r["rel_l2_velocity"] for r in rows])
    rho, pval = spearmanr(_d, _e)
    ax.set_title(f"Error broadly tracks distance from\nthe fitted lids "
                 f"(Spearman $\\rho$={rho:.2f}, n={len(_d)})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    p = FIGDIR / "fig6_error_summary.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    data = generate(verbose=False)
    fields_fn, _ = load_model()

    paths = {}
    paths["fig1"] = fig_input_functions(data)
    paths["fig2"] = fig_training_loss()
    paths["fig5"] = fig_solver_validation()

    for i in range(len(data["test_g"])):
        p, m = fig_test_summary(data, fields_fn, i)
        paths[f"fig3_test{i+1}"] = p

    paths["fig4"] = fig_operator_gallery(data, fields_fn, indices=(0, 1, 2, 3))
    paths["fig6"] = fig_error_summary()

    for k, v in paths.items():
        print(f"{k:16s} {v.relative_to(v.parents[2])}")
    return paths


if __name__ == "__main__":
    main()
