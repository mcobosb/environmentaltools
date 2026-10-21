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


def plot_map(cfg, case, grid, quantity="hs", ax=None, cmap="viridis", arrows="dir", n_arrows=20, folder=None,
             **kw):
    """Map of a result quantity for a case (from ``map_<grid>.tab``), with direction arrows.

    Args:
        n_arrows (int): Approximate number of arrows along each axis (one per spatial bin).
    """
    m = read_map(cfg, case, grid, folder)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ok = m[quantity].notna() & (m.get("depth", m[quantity]) > cfg["set"]["depmin"])
    tc = ax.tricontourf(m.x[ok], m.y[ok], m[quantity][ok], levels=20, cmap=cmap, **kw)
    plt.colorbar(tc, ax=ax, label=quantity)
    if arrows and arrows in m:
        s = m[ok & m[arrows].notna()]
        bx = np.floor((s.x - s.x.min()) / (np.ptp(s.x) / n_arrows + 1e-9))
        by = np.floor((s.y - s.y.min()) / (np.ptp(s.y) / n_arrows + 1e-9))
        s = s.groupby([bx, by]).first()
        a = np.deg2rad(270 - s[arrows])          # nautical (from) → cartesian (towards)
        ax.quiver(s.x, s.y, np.cos(a), np.sin(a), color="k", width=0.003, headwidth=4,
                  scale=1.5 * n_arrows, pivot="middle")
    ax.set_xlim(m.x[ok].min(), m.x[ok].max())
    ax.set_ylim(m.y[ok].min(), m.y[ok].max())
    ax.set_aspect("equal")
    return ax


def plot_case_maps(cfg, case, folder=None, quantity=None, file_name=None, dpi=None):
    """One panel per saved map of a case (all grids), with the output points; optionally saved to a PNG.

    Defaults come from ``outputs: figures:`` (``quantity``, ``dpi``, ``n_arrows``).
    """
    import json
    from pathlib import Path

    fs = cfg["outputs"].get("figures") or {}
    quantity = quantity or fs.get("quantity", "hs")
    grids = [g for g in cfg.grid_names if (cfg["outputs"].get("maps") or {}).get(g)]
    fig, axs = plt.subplots(1, len(grids), figsize=(7 * len(grids), 5.5), squeeze=False)
    pts = None
    if (cfg.cases_dir / "_common" / "points.csv").exists():
        pts = pd.read_csv(cfg.cases_dir / "_common" / "points.csv")
    meta_f = Path(folder or cfg.results_dir / case) / "case.json"
    v = json.loads(meta_f.read_text())["values"] if meta_f.exists() else {}
    for ax, g in zip(axs[0], grids):
        plot_map(cfg, case, g, quantity, ax=ax, folder=folder, n_arrows=fs.get("n_arrows", 20))
        if pts is not None:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            p = pts[pts.x.between(*xl) & pts.y.between(*yl)]
            ax.plot(p.x, p.y, "w^", ms=6, mec="k")
            for r in p.itertuples():
                ax.annotate(r.name, (r.x, r.y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{case} – {g}", fontsize=9)
    if v:
        fig.suptitle(f"Hs {v.get('Hs', np.nan):.2f} m, Tp {v.get('Tp', np.nan):.1f} s, Dir {v.get('Dir', np.nan):.0f}°"
                     + (f", wind {v['Vv']:.1f} m/s {v['DirV']:.0f}°" if v.get("Vv") is not None else ""), fontsize=10)
    fig.tight_layout()
    if file_name:
        fig.savefig(file_name, dpi=dpi or fs.get("dpi", 120))
        plt.close(fig)
    return fig
