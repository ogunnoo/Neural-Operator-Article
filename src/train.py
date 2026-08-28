"""Train the DeepONet on (lid function, CFD velocity field) pairs, in JAX.

Purely supervised: the loss is MSE between predicted and CFD velocities.  There
is no PDE residual term anywhere -- the physics enters only through the training
data.  That is deliberate; it is the cleanest way to show what operator
learning is, separately from what physics-informed training is.

Batching follows the outer-product structure of the model: every step uses ALL
fitting functions and a random subset of grid points.  The branch network is
therefore evaluated once per function per step rather than once per
(function, point) pair, which is what makes the whole run cheap.
"""

from __future__ import annotations

import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import CFG, MODELDIR, DATADIR
from .dataset import generate
from .deeponet import (DeepONetUV, init_params, make_norm, n_params,
                       normalise_g, save_checkpoint)


def grid_coords(x, y):
    """(M, 2) array of grid points, ordered to match field.ravel()."""
    X, Y = np.meshgrid(x, y)                      # X[j,i] = x_i, Y[j,i] = y_j
    return np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)


def stack_uv(u, v):
    """(N, M, 2) targets from (N, ny, nx) field stacks."""
    return np.stack([u.reshape(len(u), -1), v.reshape(len(v), -1)], -1).astype(np.float32)


def rel_l2_over_functions(pred_fn, params, coords, gs, us, vs):
    """Mean over functions of the combined-velocity relative L2 error."""
    uv = np.asarray(pred_fn(params, jnp.asarray(gs, jnp.float32), coords))
    errs = []
    for k, (u, v) in enumerate(zip(us, vs)):
        up = uv[k, :, 0].reshape(u.shape)
        vp = uv[k, :, 1].reshape(v.shape)
        num = np.sqrt(((up - u) ** 2 + (vp - v) ** 2).sum())
        den = np.sqrt((u**2 + v**2).sum())
        errs.append(num / den)
    return float(np.mean(errs))


def main(verbose: bool = True, save: bool = True, tag: str = ""):
    data = generate(verbose=verbose)
    x, y = data["x"], data["y"]

    g_all, u_all, v_all = data["train_g"], data["train_u"], data["train_v"]
    n_fit = len(g_all) - CFG.n_val
    fit, val = slice(0, n_fit), slice(n_fit, len(g_all))

    coords_np = grid_coords(x, y)
    coords = jnp.asarray(coords_np)
    M = coords_np.shape[0]

    G_fit = g_all[fit].astype(np.float32)
    UV_fit = stack_uv(u_all[fit], v_all[fit])                # (N, M, 2)
    norm = make_norm(G_fit, UV_fit.reshape(-1, 2))

    model = DeepONetUV(
        latent=CFG.latent,
        branch_hidden=CFG.branch_hidden, branch_depth=CFG.branch_depth,
        trunk_hidden=CFG.trunk_hidden, trunk_depth=CFG.trunk_depth,
    )
    key = jax.random.PRNGKey(CFG.seed)
    key, init_key = jax.random.split(key)
    params = init_params(model, init_key, CFG.n_sensors)

    Gz = jnp.asarray(normalise_g(G_fit, norm))               # (N, m)
    UVn = jnp.asarray((UV_fit - norm["uv_mean"]) / norm["uv_std"])

    if verbose:
        print(f"[train] backend={jax.default_backend()} devices={jax.devices()}")
        print(f"[train] fit functions={n_fit}  val functions={CFG.n_val}  "
              f"grid points={M}  supervised targets={n_fit*M*2:,}")
        print(f"[train] steps={CFG.steps:,}  points/step={CFG.points_per_step}")
        print(f"[train] DeepONet parameters: {n_params(params):,}")

    warmup = int(CFG.warmup_frac * CFG.steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=CFG.lr * 0.05, peak_value=CFG.lr,
        warmup_steps=warmup, decay_steps=CFG.steps, end_value=CFG.lr_final,
    )
    # Gradient clipping removes the mid-training loss spikes that a plain Adam
    # run on this problem is prone to; the small decoupled weight decay is the
    # main thing holding back overfitting to a few dozen training functions.
    opt = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adamw(schedule, weight_decay=CFG.weight_decay),
    )
    opt_state = opt.init(params)

    def loss_fn(p, idx):
        pred = model.apply({"params": p}, Gz, coords[idx])   # (N, |idx|, 2)
        return jnp.mean((pred - UVn[:, idx]) ** 2)

    @jax.jit
    def pred_fn(p, gs, cc):
        uvn = model.apply({"params": p}, normalise_g(gs, norm), cc)
        return uvn * norm["uv_std"] + norm["uv_mean"]

    @partial(jax.jit, donate_argnums=(0, 1), static_argnums=(3,))
    def chunk(params, opt_state, key, n):
        """`n` optimisation steps fused into one XLA call."""
        def step(carry, k):
            p, os_ = carry
            idx = jax.random.choice(k, M, (CFG.points_per_step,), replace=False)
            loss, grads = jax.value_and_grad(loss_fn)(p, idx)
            updates, os_ = opt.update(grads, os_, p)
            p = optax.apply_updates(p, updates)
            return (p, os_), loss

        keys = jax.random.split(key, n)
        (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), keys)
        return params, opt_state, jnp.mean(losses)

    CHUNK = 250                       # steps between validation checks
    history, hist_steps, val_history, val_steps = [], [], [], []
    best_val, best_params, best_step = float("inf"), None, -1
    t0 = time.time()

    for c in range(CFG.steps // CHUNK):
        key, ckey = jax.random.split(key)
        params, opt_state, loss = chunk(params, opt_state, ckey, CHUNK)
        done = (c + 1) * CHUNK
        history.append(float(loss))
        hist_steps.append(done)

        vl = rel_l2_over_functions(
            pred_fn, params, coords, g_all[val], u_all[val], v_all[val]
        )
        val_history.append(vl)
        val_steps.append(done)
        if vl < best_val:
            best_val, best_step = vl, done
            best_params = jax.tree_util.tree_map(np.array, params)

        if verbose and (done % 2500 == 0 or done == CHUNK):
            print(f"[train] step {done:6d}  mse(norm) {float(loss):.4e}  "
                  f"val relL2 {vl:.4%}  lr {float(schedule(done)):.2e}  "
                  f"{time.time()-t0:.0f}s")

    train_seconds = time.time() - t0
    final_params = best_params if best_params is not None else params

    ckpt = MODELDIR / f"deeponet_cavity{tag}.msgpack"
    if save:
        save_checkpoint(ckpt, final_params, norm, {
            "latent": CFG.latent, "branch_hidden": CFG.branch_hidden,
            "branch_depth": CFG.branch_depth, "trunk_hidden": CFG.trunk_hidden,
            "trunk_depth": CFG.trunk_depth, "n_sensors": CFG.n_sensors,
        })

    hist = dict(
        train_mse=history, train_steps=hist_steps,
        val_rel_l2=val_history, val_steps=val_steps,
        best_val_rel_l2=best_val, best_step=best_step,
        train_seconds=train_seconds, backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        n_params=n_params(final_params), n_fit_functions=int(n_fit),
        n_val_functions=int(CFG.n_val), grid_points=int(M),
        steps=CFG.steps, points_per_step=CFG.points_per_step,
        lr=CFG.lr, lr_final=CFG.lr_final, weight_decay=CFG.weight_decay,
        grad_clip=CFG.grad_clip, seed=CFG.seed,
    )
    if save:
        (DATADIR / f"training_history{tag}.json").write_text(json.dumps(hist, indent=2))

    if verbose:
        print(f"[train] done in {train_seconds:.0f}s; best val relL2 "
              f"{best_val:.4%} at step {best_step}; saved {ckpt.name}")
    return hist


if __name__ == "__main__":
    main()
