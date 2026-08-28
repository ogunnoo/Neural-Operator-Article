"""Steady 2D lid-driven cavity solver (streamfunction-vorticity formulation).

Why streamfunction-vorticity instead of a primitive-variable projection method?

  u = d(psi)/dy ,  v = -d(psi)/dx

means the discrete velocity field is divergence-free *by construction* -- there
is no pressure-Poisson iteration that can be left half-converged and silently
pollute the "reference" data.  Since these fields are the ground truth the
neural operator is asked to reproduce, that structural guarantee matters more
here than having a general-purpose CFD code.

Governing equations (incompressible, constant nu = 1/Re, U_ref = L = 1):

    vorticity transport   dw/dt + u dw/dx + v dw/dy = nu * lap(w)
    kinematics            lap(psi) = -w

Boundary conditions: psi = 0 on all four walls (no flow through them), and
Thom's first-order wall-vorticity condition supplies w on the walls.  For the
top wall moving with tangential velocity g(x):

    w_wall = -2 * psi_{one node in} / h^2  -  2 * g(x) / h

The equations are marched in pseudo-time with explicit Euler and second-order
central differences until dw/dt falls below a tolerance, i.e. until the
solution is genuinely steady rather than merely "run for a fixed number of
steps".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# Poisson operator for the streamfunction (built once, factorised once)
# --------------------------------------------------------------------------

_LU_CACHE: dict[int, object] = {}


def _psi_solver(n: int):
    """Return a prefactorised solver for lap(psi) = -w with psi = 0 on walls.

    The interior 5-point Laplacian is the same matrix for every lid profile and
    every pseudo-time step, so we factorise it once and reuse the LU factors.
    That turns the kinematic step into an exact (to machine precision) solve
    instead of a truncated iterative sweep.
    """
    if n in _LU_CACHE:
        return _LU_CACHE[n]

    m = n - 2                                    # interior nodes per direction
    main = sp.diags([1.0, -4.0, 1.0], [-1, 0, 1], shape=(m, m), format="csr")
    off = sp.identity(m, format="csr")
    I = sp.identity(m, format="csr")
    # 2D Laplacian on the interior with homogeneous Dirichlet data
    A = sp.kron(I, main) + sp.kron(sp.diags([1.0, 1.0], [-1, 1], shape=(m, m)), off)
    lu = spla.splu(A.tocsc())
    _LU_CACHE[n] = lu
    return lu


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------

def velocity_from_psi(psi: np.ndarray, h: float, lid: np.ndarray):
    """u = d(psi)/dy, v = -d(psi)/dx, with the exact wall values imposed."""
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)

    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * h)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * h)

    # Dirichlet velocity data on the walls (no-slip, plus the moving lid).
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = lid
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


def _wall_vorticity(w: np.ndarray, psi: np.ndarray, h: float, lid: np.ndarray) -> None:
    """Thom's condition on all four walls (in place)."""
    w[-1, :] = -2.0 * psi[-2, :] / h**2 - 2.0 * lid / h      # moving top lid
    w[0, :] = -2.0 * psi[1, :] / h**2                        # stationary floor
    w[:, 0] = -2.0 * psi[:, 1] / h**2                        # left wall
    w[:, -1] = -2.0 * psi[:, -2] / h**2                      # right wall
    # Corner nodes never enter an interior stencil; zero them for clean plots.
    w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0.0


# --------------------------------------------------------------------------
# Main solver
# --------------------------------------------------------------------------

@dataclass
class CavitySolution:
    u: np.ndarray
    v: np.ndarray
    psi: np.ndarray
    w: np.ndarray
    steps: int
    residual: float          # final ||dw/dt||_inf
    converged: bool
    max_divergence: float    # discrete |div u|_inf on the interior


def solve_cavity(
    lid: np.ndarray,
    re: float = 100.0,
    n: int = 41,
    cfl: float = 0.35,
    tol: float = 1e-6,
    max_steps: int = 200_000,
) -> CavitySolution:
    """Solve the steady cavity flow driven by the lid profile `lid` (length n).

    Array convention: field[j, i] with j indexing y and i indexing x, both on
    a uniform grid over [0, 1].
    """
    assert lid.shape == (n,), f"lid must have length {n}, got {lid.shape}"

    h = 1.0 / (n - 1)
    nu = 1.0 / re
    lu = _psi_solver(n)

    psi = np.zeros((n, n))
    w = np.zeros((n, n))

    # Explicit-Euler stability: diffusive limit h^2/(4 nu) and advective limit
    # h / |u|max.  Recomputed periodically as the flow spins up.
    u_scale = max(float(np.max(np.abs(lid))), 1e-3)
    dt = cfl * min(h * h / (4.0 * nu), h / u_scale)

    residual = np.inf
    converged = False
    step = 0

    for step in range(1, max_steps + 1):
        # --- kinematics: solve lap(psi) = -w exactly on the interior ---
        rhs = (-h * h * w[1:-1, 1:-1]).ravel()
        psi[1:-1, 1:-1] = lu.solve(rhs).reshape(n - 2, n - 2)

        u, v = velocity_from_psi(psi, h, lid)
        _wall_vorticity(w, psi, h, lid)

        # --- vorticity transport, central differences on the interior ---
        wc = w[1:-1, 1:-1]
        dwdx = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * h)
        dwdy = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * h)
        lap_w = (
            w[1:-1, 2:] + w[1:-1, :-2] + w[2:, 1:-1] + w[:-2, 1:-1] - 4.0 * wc
        ) / (h * h)

        dwdt = -u[1:-1, 1:-1] * dwdx - v[1:-1, 1:-1] * dwdy + nu * lap_w
        w[1:-1, 1:-1] = wc + dt * dwdt

        if step % 50 == 0:
            residual = float(np.max(np.abs(dwdt)))
            if not np.isfinite(residual) or residual > 1e12:
                raise FloatingPointError(
                    f"cavity solver diverged at step {step} (residual={residual:.3e}); "
                    "reduce cfl or refine the grid"
                )
            if residual < tol:
                converged = True
                break
            # keep the time step honest as the interior velocities grow
            umax = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1e-3)
            dt = cfl * min(h * h / (4.0 * nu), h / umax)

    # Final consistent state
    rhs = (-h * h * w[1:-1, 1:-1]).ravel()
    psi[1:-1, 1:-1] = lu.solve(rhs).reshape(n - 2, n - 2)
    u, v = velocity_from_psi(psi, h, lid)
    _wall_vorticity(w, psi, h, lid)

    div = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
        + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
    )

    return CavitySolution(
        u=u,
        v=v,
        psi=psi,
        w=w,
        steps=step,
        residual=residual,
        converged=converged,
        max_divergence=float(np.max(np.abs(div))),
    )


def grid(n: int):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    return x, y
