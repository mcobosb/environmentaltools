"""Figures of a SWAN project: grids, boundaries and output points; result maps."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cases import boundary_distances
from .read import read_map


def plot_grids(cfg, ax=None, every=10, depth=True):
    """Grids (outline and every ``every``-th grid line), forced boundaries and output points.

    Args:
        cfg (SwanConfig): Configuration.
        every (int): Draw one grid line every ``every`` nodes.
        depth (bool): Shade the depth of the first (outer) grid.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    grids = cfg.grids()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, name in enumerate(cfg.grid_names):
        g, c = grids[name], colors[k % len(colors)]
        if depth and k == 0 and g.depth is not None:
            pc = ax.pcolormesh(g.x, g.y, np.where(g.depth > cfg["set"]["depmin"], g.depth, np.nan),
                               cmap="Blues", shading="auto")
            plt.colorbar(pc, ax=ax, label="depth (m)")
        for jj in range(0, g.shape[0], every):
            ax.plot(g.x[jj], g.y[jj], color=c, lw=0.3, alpha=0.6)
        for ii in range(0, g.shape[1], every):
            ax.plot(g.x[:, ii], g.y[:, ii], color=c, lw=0.3, alpha=0.6)
        j, i = g.perimeter()
        ax.plot(np.r_[g.x[j, i], g.x[0, 0]], np.r_[g.y[j, i], g.y[0, 0]], color=c, lw=1.2, label=name)
        for b, spec in enumerate(cfg["grids"][name].get("boundaries") or []):
            if "segment_ij" in spec:
                d = boundary_distances(cfg, name, b)
                ax.plot(d.x, d.y, color="r", lw=3, alpha=0.7)
    pts = cfg["outputs"].get("points")
    if pts is not None:
        p = pd.read_csv(cfg.path(pts)) if isinstance(pts, str) else pd.DataFrame(pts)
        ax.plot(p.x, p.y, "k^", ms=5)
        for r in p.itertuples():
            ax.annotate(getattr(r, "name", ""), (r.x, r.y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    return ax


def plot_map(cfg, case, grid, quantity="hs", ax=None, cmap="viridis", arrows="dir", every=3, **kw):
    """Map of a result quantity for a case (from ``map_<grid>.tab``), with direction arrows."""
    m = read_map(cfg, case, grid)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ok = m[quantity].notna()
    tc = ax.tricontourf(m.x[ok], m.y[ok], m[quantity][ok], levels=20, cmap=cmap, **kw)
    plt.colorbar(tc, ax=ax, label=quantity)
    if arrows and arrows in m:
        s = m[ok].iloc[::every]
        a = np.deg2rad(270 - s[arrows])          # nautical (from) → cartesian (towards)
        ax.quiver(s.x, s.y, np.cos(a), np.sin(a), color="k", scale=40, width=0.002)
    ax.set_aspect("equal")
    ax.set_title(f"{case} – {grid}")
    return ax
