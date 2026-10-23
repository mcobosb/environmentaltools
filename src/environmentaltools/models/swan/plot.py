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


def _map_shape(cfg, grid):
    """Node shape ``(ny, nx)`` of the map of ``grid`` (whole grid or subgrid)."""
    sub = ((cfg["outputs"].get("maps") or {}).get(grid) or {}).get("subgrid")
    if sub:
        i1, i2, j1, j2 = sub
        return j2 - j1 + 1, i2 - i1 + 1
    return cfg.grids()[grid].shape


def _km_axes(ax):
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)
    ax.set_xlabel("x (km)", fontsize=8)
    ax.set_ylabel("y (km)", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_map(cfg, case, grid, quantity="hs", ax=None, cmap="viridis", arrows="dir", n_arrows=20, folder=None,
             levels=20, **kw):
    """Map of a result quantity for a case (from ``map_<grid>.tab``), with direction arrows.

    The values are drawn on the structured (curvilinear or regular) grid, so dry nodes
    and land stay blank.

    Args:
        n_arrows (int): Approximate number of arrows along each axis (one per spatial bin).
    """
    m = read_map(cfg, case, grid, folder)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    wet = m[quantity].notna()
    shape = _map_shape(cfg, grid)
    structured = len(m) == shape[0] * shape[1]
    if structured:
        # land = nodes dry in the INPUT bathymetry (output DEPTH includes the wave set-up, so land
        # nodes stored at DEPMIN can appear slightly wet with Hs ≈ 0)
        dep = cfg.grids()[grid].depth
        sub = ((cfg["outputs"].get("maps") or {}).get(grid) or {}).get("subgrid")
        if sub:
            i1, i2, j1, j2 = sub
            dep = dep[j1:j2 + 1, i1:i2 + 1]
        wet &= pd.Series(np.nan_to_num(dep, nan=0).ravel() > cfg["set"]["depmin"], index=m.index)
    elif "depth" in m:
        wet &= m.depth > cfg["set"]["depmin"]
    if structured:
        X, Y = m.x.to_numpy().reshape(shape), m.y.to_numpy().reshape(shape)
        Z = np.ma.masked_where(~wet.to_numpy().reshape(shape), m[quantity].to_numpy().reshape(shape))
        tc = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, **kw)
    else:                                                      # unknown layout: scattered nodes
        tc = ax.tricontourf(m.x[wet], m.y[wet], m[quantity][wet], levels=levels, cmap=cmap, **kw)
    plt.colorbar(tc, ax=ax, label=quantity, shrink=0.8)
    if arrows and arrows in m:
        s = m[wet & m[arrows].notna()]
        bx = np.floor((s.x - s.x.min()) / (np.ptp(s.x) / n_arrows + 1e-9))
        by = np.floor((s.y - s.y.min()) / (np.ptp(s.y) / n_arrows + 1e-9))
        s = s.groupby([bx, by]).first()
        a = np.deg2rad(270 - s[arrows])          # nautical (from) → cartesian (towards)
        ax.quiver(s.x, s.y, np.cos(a), np.sin(a), color="k", width=0.0025, headwidth=4,
                  scale=1.6 * n_arrows, pivot="middle")
    ax.set_xlim(m.x[wet].min(), m.x[wet].max())
    ax.set_ylim(m.y[wet].min(), m.y[wet].max())
    ax.set_aspect("equal")
    _km_axes(ax)
    return ax


def plot_wave_field(cfg, case, grid, quantity="hs", folder=None, depth_min=1.0, **kw):
    """Publication map of a case: same figure as :func:`environmentaltools.models.delft3d.plot_wave_field`.

    Reads ``map_<grid>.tab`` (see :func:`read_map`) and draws it with
    :func:`environmentaltools.models.delft3d.plot_wave_field_grid` (cartopy UTM
    basemap, coastline, land fill, arrows scaled by the field), so Delft3D and
    SWAN figures look the same. Land and dry nodes are taken from the INPUT
    bathymetry; nodes shallower than ``depth_min`` are hidden, as in Delft3D.

    Args:
        cfg (SwanConfig): Configuration.
        case (str): Case name.
        grid (str): Grid name (its map must be a whole grid, not a subgrid).
        quantity (str): Column of the map to draw (``hs``, ``tps``…).
        folder (str | Path | None): Folder with ``map_<grid>.tab`` (else
            ``results_dir/<case>`` or ``archive_dir/<case>``).
        depth_min (float): Hide nodes shallower than this (m). 0/None = only dry nodes.
        **kw: Passed to ``plot_wave_field_grid`` (``coast_xyz``, ``cmap``, ``vmin``,
            ``vmax``, ``quiver_stride``, ``quiver_scale``, ``quiver_color``, ``title``,
            ``fname``…). Arrows use the map column ``dir`` when present
            (``directions=None`` to switch them off).

    Returns:
        matplotlib.axes.GeoAxes: The cartopy axes.
    """
    from ..delft3d.plot import plot_wave_field_grid

    m = read_map(cfg, case, grid, folder)
    shape = _map_shape(cfg, grid)
    if len(m) != shape[0] * shape[1] or ((cfg["outputs"].get("maps") or {}).get(grid) or {}).get("subgrid"):
        raise ValueError(f"map_{grid}.tab of {case} is not a whole-grid map {shape}")
    g = cfg.grids()[grid]
    dep = np.nan_to_num(g.depth, nan=0.0)
    hide = ~(dep > max(depth_min or 0.0, cfg["set"]["depmin"]))
    # node coordinates from the grid: XP/YP of a TABLE keep only ~4 significant digits
    # (0.5383E+06 → ±50 m in x, ±500 m in y in UTM), which turns the grid edges into steps
    X, Y = g.x, g.y
    val = np.where(hide, np.nan, m[quantity].to_numpy().reshape(shape))
    if "directions" not in kw and "dir" in m:
        kw["directions"] = m["dir"].to_numpy().reshape(shape)     # nautical (SET NAUTICAL)
    variable = {"hs": "hsign"}.get(quantity, quantity)
    kw.setdefault("title", "")
    return plot_wave_field_grid(X, Y, val, variable=variable, **kw)


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
    metas = [Path(folder) / "case.json"] if folder else []
    metas += [cfg.results_dir / case / "case.json", cfg.cases_dir / case / "case.json"]
    meta_f = next((f for f in metas if f.exists()), None)
    v = json.loads(meta_f.read_text())["values"] if meta_f else {}
    vmax = max(np.nanmax(read_map(cfg, case, g, folder)[quantity].to_numpy()) for g in grids)
    levels = np.linspace(0, vmax if vmax > 0 else 1, 21)        # same colour scale in every panel
    for ax, g in zip(axs[0], grids):
        plot_map(cfg, case, g, quantity, ax=ax, folder=folder, n_arrows=fs.get("n_arrows", 20), levels=levels)
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
