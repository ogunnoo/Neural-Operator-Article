"""DeepONet in JAX/Flax for the map  g(x)  ->  [u(x,y), v(x,y)].

The whole point of the branch/trunk split is that it separates *which problem*
from *where in the domain*:

    branch(g)   encodes the entire input function into p coefficients
    trunk(x, y) evaluates p basis functions at a query point
    output      = (1/sqrt(p)) * sum_k branch_k(g) * trunk_k(x, y)  +  bias

The trunk learns a set of global basis functions over the cavity; the branch
learns how much of each basis function a given lid profile calls for.  That is
a learned, nonlinear analogue of expanding the solution in a basis whose
coefficients are functionals of the input function.

We use two independent (branch, trunk) latent pairs -- one for u and one for v
-- because the two components have quite different spatial structure.

Everything the network sees is normalised; `predict` wraps the raw module so
callers can pass physical lid profiles and get physical velocities back.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization


class MLP(nn.Module):
    """Plain tanh MLP. Smooth activations suit smooth PDE solution fields."""
    out_dim: int
    hidden: int
    depth: int

    @nn.compact
    def __call__(self, z):
        for _ in range(self.depth):
            z = nn.tanh(nn.Dense(self.hidden)(z))
        return nn.Dense(self.out_dim)(z)


class DeepONetUV(nn.Module):
    latent: int = 128
    branch_hidden: int = 160
    branch_depth: int = 4
    trunk_hidden: int = 160
    trunk_depth: int = 4

    @nn.compact
    def __call__(self, g_norm, xy):
        """Evaluate the operator for N input functions at M query points.

        g_norm : (N, m)  normalised lid samples -- N whole input functions
        xy     : (M, 2)  query coordinates in the unit square
        returns: (N, M, 2) normalised (u, v)

        Note the shape of the computation: the branch runs once per *function*
        and the trunk once per *point*, and they meet in an outer product.  The
        trunk output t_k(x, y) is a learned basis over the cavity; the branch
        output b_k(g) is the coefficient that this particular lid profile puts
        on basis function k.  This is exactly the DeepONet ansatz

            u(x, y) ~ sum_k b_k(g) * t_k(x, y)

        and writing it this way is also what makes training cheap: the branch
        is not re-evaluated once per collocation point.
        """
        # Map the unit square to [-1, 1], the natural input range for tanh nets.
        t_in = 2.0 * xy - 1.0

        b = MLP(2 * self.latent, self.branch_hidden, self.branch_depth)(g_norm)
        t = MLP(2 * self.latent, self.trunk_hidden, self.trunk_depth)(t_in)

        b_u, b_v = jnp.split(b, 2, axis=-1)          # (N, p) each
        t_u, t_v = jnp.split(t, 2, axis=-1)          # (M, p) each

        bias = self.param("bias", nn.initializers.zeros, (2,))
        # 1/sqrt(p) keeps the dot product O(1) at initialisation.
        scale = self.latent ** -0.5
        u = jnp.einsum("np,mp->nm", b_u, t_u) * scale + bias[0]
        v = jnp.einsum("np,mp->nm", b_v, t_v) * scale + bias[1]
        return jnp.stack([u, v], axis=-1)            # (N, M, 2)


# --------------------------------------------------------------------------
# Normalisation + physical-units wrapper
# --------------------------------------------------------------------------

def make_norm(g_fit: np.ndarray, uv_fit: np.ndarray) -> dict:
    """Normalisation statistics, computed on the fitting split only."""
    return {
        "g_mean": np.asarray(g_fit.mean(0), np.float32),
        "g_std": np.asarray(g_fit.std(0) + 1e-3, np.float32),
        "uv_mean": np.asarray(uv_fit.mean(0), np.float32),
        "uv_std": np.asarray(uv_fit.std(0) + 1e-8, np.float32),
    }


def normalise_g(g, norm):
    return (g - norm["g_mean"]) / norm["g_std"]


def predict(model: DeepONetUV, params, norm, g, xy):
    """Physical-units (u, v).

    `g` may be (m,) for a single input function or (N, m) for a batch; `xy` is
    (M, 2).  Returns (M, 2) or (N, M, 2) correspondingly.
    """
    single = (g.ndim == 1)
    gz = normalise_g(jnp.atleast_2d(g), norm)
    uvn = model.apply({"params": params}, gz, xy)
    out = uvn * norm["uv_std"] + norm["uv_mean"]
    return out[0] if single else out


def init_params(model: DeepONetUV, key, n_sensors: int):
    dummy_g = jnp.zeros((1, n_sensors))
    dummy_xy = jnp.zeros((1, 2))
    return model.init(key, dummy_g, dummy_xy)["params"]


def n_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# --------------------------------------------------------------------------
# Checkpointing (params + normalisation in one self-contained file)
# --------------------------------------------------------------------------

def save_checkpoint(path, params, norm, cfg_dict: dict[str, Any]):
    blob = serialization.to_bytes({"params": params, "norm": norm})
    path.write_bytes(blob)
    path.with_suffix(".json").write_text(__import__("json").dumps(cfg_dict, indent=2))


def load_checkpoint(path, n_sensors: int):
    import json

    cfg_dict = json.loads(path.with_suffix(".json").read_text())
    model = DeepONetUV(
        latent=cfg_dict["latent"],
        branch_hidden=cfg_dict["branch_hidden"],
        branch_depth=cfg_dict["branch_depth"],
        trunk_hidden=cfg_dict["trunk_hidden"],
        trunk_depth=cfg_dict["trunk_depth"],
    )
    template_params = init_params(model, jax.random.PRNGKey(0), n_sensors)
    template_norm = {
        "g_mean": np.zeros(n_sensors, np.float32),
        "g_std": np.ones(n_sensors, np.float32),
        "uv_mean": np.zeros(2, np.float32),
        "uv_std": np.ones(2, np.float32),
    }
    restored = serialization.from_bytes(
        {"params": template_params, "norm": template_norm}, path.read_bytes()
    )
    return model, restored["params"], restored["norm"], cfg_dict
