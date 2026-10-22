"""Figures of Delft3D grids and DELFT3D-WAVE map fields on a cartopy UTM basemap."""

import copy
from pathlib import Path

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize

from ..common.grids import read_delft_grd
from .read import _NEFIS_FILL, _delft_grid_params, _delft_read_coords, _delft_read_var


def _cartopy_crs():
    """Import ``cartopy.crs`` on demand (cartopy can fail to import, e.g. under
    the debugger with Python 3.13)."""
    try:
        import cartopy.crs as ccrs
    except (ImportError, AssertionError, KeyError) as e:
        raise ImportError(
            "cartopy is required for the Delft3D map figures. "
            "Install it with: conda install -c conda-forge cartopy"
        ) from e
    return ccrs


def _build_grid_segments(X, Y):
    """Build all cell-edge segments for a DELFT3D curvilinear grid.

    Args:
        X (np.ndarray): 2-D array of X coordinates (nmax, mmax).
        Y (np.ndarray): 2-D array of Y coordinates (nmax, mmax).

    Returns:
        np.ndarray: Array of shape (N, 2, 2) with [start, end] pairs for
        every valid grid edge, ready for use with ``LineCollection``.
    """
    x1h, x2h = X[:, :-1], X[:, 1:]
    y1h, y2h = Y[:, :-1], Y[:, 1:]
    valid_h = ~(np.isnan(x1h) | np.isnan(x2h))
    segs_h = np.stack(
        [np.stack([x1h, y1h], axis=-1), np.stack([x2h, y2h], axis=-1)], axis=-2
    )[valid_h]

    x1v, x2v = X[:-1, :], X[1:, :]
    y1v, y2v = Y[:-1, :], Y[1:, :]
    valid_v = ~(np.isnan(x1v) | np.isnan(x2v))
    segs_v = np.stack(
        [np.stack([x1v, y1v], axis=-1), np.stack([x2v, y2v], axis=-1)], axis=-2
    )[valid_v]

    return np.vstack([segs_h, segs_v])


def _load_coast_segments(coast_path, gap_m=5000):
    """Load an XYZ coastline file and split it into continuous segments.

    The file must be a CSV with ``x`` and ``y`` columns in the same CRS as
    the grids. Segments are separated wherever consecutive points are more
    than *gap_m* metres apart.

    Args:
        coast_path (str or Path): Path to the CSV coastline file.
        gap_m (float): Gap threshold in metres. Defaults to 5000.

    Returns:
        list[np.ndarray]: List of (N, 2) arrays, one per segment.
    """
    import pandas as _pd

    df = _pd.read_csv(coast_path)
    xy = df[["x", "y"]].values
    dist = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    breaks = list(np.where(dist > gap_m)[0] + 1)
    segs, prev = [], 0
    for b in breaks + [len(xy)]:
        segs.append(xy[prev:b])
        prev = b
    return segs


def _build_land_polygon(coast_xy, xmin, xmax, ymin, ymax,
                        land_is_north=True, pad=20000):
    """Build a filled land polygon from an open coastline.

    The coastline must run roughly west–east across the domain. The polygon
    is closed through the *north* (high-Y) edge of the padded extent when
    ``land_is_north=True``, or through the *south* edge otherwise.

    Args:
        coast_xy (np.ndarray): Array of shape (N, 2) with the main coastline.
        xmin, xmax, ymin, ymax (float): Map extent in metres.
        land_is_north (bool): If True, land is north of the coast (default).
        pad (float): Extra padding beyond the extent in metres. Defaults to 20000.

    Returns:
        np.ndarray or None: Array of polygon vertices (M, 2), or None if the
        coastline does not intersect the padded extent.
    """
    p = pad
    mask = (
        (coast_xy[:, 0] >= xmin - p) & (coast_xy[:, 0] <= xmax + p) &
        (coast_xy[:, 1] >= ymin - p) & (coast_xy[:, 1] <= ymax + p)
    )
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    pts = coast_xy[idx[0]:idx[-1] + 1]
    # Ensure west-to-east ordering
    if pts[0, 0] > pts[-1, 0]:
        pts = pts[::-1]
    close_y = ymax + p if land_is_north else ymin - p
    poly = np.vstack([
        [[xmin - p, pts[0, 1]]],
        pts,
        [[xmax + p, pts[-1, 1]]],
        [[xmax + p, close_y]],
        [[xmin - p, close_y]],
    ])
    return poly


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

_DELFT_DEFAULT_STYLES = [
    {"color": "#1565C0", "lw": 0.30, "alpha": 0.65},
    {"color": "#E65100", "lw": 0.40, "alpha": 0.85},
    {"color": "#2E7D32", "lw": 0.35, "alpha": 0.75},
    {"color": "#6A1B9A", "lw": 0.35, "alpha": 0.75},
]


def plot_grids(
    grd_paths,
    coast_xyz=None,
    points=None,
    utm_zone=30,
    margin=8000,
    color_land="#EDE8DF",
    color_ocean="#C8DCF0",
    coast_gap_m=5000,
    land_is_north=True,
    grid_styles=None,
    point_label_col=None,
    point_label_offset=500,
    title=None,
    figsize=(12, 10),
    fname=None,
):
    """Plot DELFT3D-WAVE curvilinear grids with an optional basemap.

    Draws one or more DELFT3D curvilinear grids read from ASCII ``.grd``
    files (RGFGRID format) over a basemap built from a high-resolution
    coastline polyline. Extraction or control points can be overlaid
    optionally.

    All spatial data (grids, coastline and points) must be in the same
    UTM coordinate system.

    Args:
        grd_paths (dict[str, str | Path]): Mapping of grid label → path to
            the ``.grd`` file (e.g. ``{"ext": "Alboran_ext.grd", "int":
            "Alboran_int.grd"}``). Grids are drawn in dict order.
        coast_xyz (str | Path | None): Path to a CSV file with ``x`` and
            ``y`` columns (same UTM CRS) representing the coastline as an
            ordered polyline. When provided, land and ocean are filled with
            *color_land* and *color_ocean* respectively. Defaults to None.
        points (pd.DataFrame | None): Optional table of control or
            extraction points with at least ``x`` and ``y`` columns (same
            UTM CRS). An extra column named *point_label_col* is used for
            annotations when provided. Defaults to None.
        utm_zone (int): UTM zone number used for the cartopy projection.
            Defaults to 30.
        margin (float): Map margin around the outermost grid extent, in
            metres. Defaults to 8000.
        color_land (str): Fill colour for land areas. Defaults to
            ``"#EDE8DF"``.
        color_ocean (str): Fill colour for ocean background. Defaults to
            ``"#C8DCF0"``.
        coast_gap_m (float): Threshold in metres for splitting the
            coastline polyline into separate segments. Defaults to 5000.
        land_is_north (bool): If True, the land polygon is closed through
            the northern edge of the map extent. Set to False for domains
            where land is to the south of the coastline. Defaults to True.
        grid_styles (dict[str, dict] | None): Per-grid style overrides. Keys
            must match those in *grd_paths*. Each value is a dict with any
            of ``color``, ``lw``, ``alpha``. Defaults to None (built-in
            palette).
        point_label_col (str | None): Column in *points* to use as point
            labels. If None, points are drawn without text labels. Defaults
            to None.
        point_label_offset (float): Label offset in metres from the point
            position. Defaults to 500.
        title (str | None): Figure title. Defaults to None.
        figsize (tuple): Figure size in inches. Defaults to (12, 10).
        fname (str | None): Output file path. If None, the figure is
            displayed interactively. Defaults to None.

    Returns:
        matplotlib.axes.GeoAxes: The cartopy axes with all elements drawn.

    Examples:
        >>> from pathlib import Path
        >>> from environmentaltools.models import delft3d
        >>> delft3d.plot_grids(
        ...     grd_paths={"ext": "Alboran_ext.grd", "int": "Alboran_int.grd"},
        ...     coast_xyz="linea_costa_oficial_utm30n.xyz",
        ...     points=df_points,           # DataFrame with x, y, label columns
        ...     point_label_col="label",
        ...     utm_zone=30,
        ...     title="DELFT3D grids — Alboran Sea",
        ...     fname="mallas_puntos.png",
        ... )
    """
    ccrs = _cartopy_crs()
    from environmentaltools.graphics.utils import show

    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches

    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    # ------------------------------------------------------------------
    # Ocean background
    # ------------------------------------------------------------------
    ax.set_facecolor(color_ocean)

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    legend_handles = []
    x_all, y_all = [], []

    for i, (name, path) in enumerate(grd_paths.items()):
        X, Y, _ = read_delft_grd(path)
        nmax, mmax = X.shape
        segs = _build_grid_segments(X, Y)

        # Resolve style
        default_st = _DELFT_DEFAULT_STYLES[i % len(_DELFT_DEFAULT_STYLES)]
        if grid_styles and name in grid_styles:
            st = {**default_st, **grid_styles[name]}
        else:
            st = default_st

        lc = LineCollection(
            segs,
            colors=st["color"],
            linewidths=st["lw"],
            alpha=st["alpha"],
            transform=proj,
            zorder=4,
        )
        ax.add_collection(lc)
        legend_handles.append(
            mpatches.Patch(
                facecolor=st["color"],
                alpha=0.6,
                edgecolor=st["color"],
                label=f"{name}  ({mmax}\u00d7{nmax})",
            )
        )
        x_all += [np.nanmin(X), np.nanmax(X)]
        y_all += [np.nanmin(Y), np.nanmax(Y)]

    # ------------------------------------------------------------------
    # Map extent
    # ------------------------------------------------------------------
    extent_utm = [
        min(x_all) - margin, max(x_all) + margin,
        min(y_all) - margin, max(y_all) + margin,
    ]
    ax.set_extent(extent_utm, crs=proj)
    xmin, xmax, ymin, ymax = extent_utm

    # ------------------------------------------------------------------
    # Coastline and land fill
    # ------------------------------------------------------------------
    if coast_xyz is not None:
        coast_segs = _load_coast_segments(coast_xyz, gap_m=coast_gap_m)

        # Merge the two longest open segments into the main coastline
        open_idx = [i for i, s in enumerate(coast_segs)
                    if np.hypot(s[-1, 0] - s[0, 0], s[-1, 1] - s[0, 1]) > coast_gap_m]
        closed_idx = [i for i in range(len(coast_segs)) if i not in open_idx]
        open_segs = [coast_segs[i] for i in open_idx]
        closed_segs = [coast_segs[i] for i in closed_idx]

        if len(open_segs) >= 2:
            # Find and merge the pair that share an endpoint
            s0, s1 = open_segs[0], open_segs[1]
            if np.allclose(s0[-1], s1[0], atol=10):
                main_coast = np.vstack([s0, s1[1:]])
            elif np.allclose(s1[-1], s0[0], atol=10):
                main_coast = np.vstack([s1, s0[1:]])
            else:
                # Fallback: concatenate in order and sort by x range
                main_coast = np.vstack(open_segs)
        elif len(open_segs) == 1:
            main_coast = open_segs[0]
        else:
            main_coast = np.vstack(coast_segs)

        # Land polygon from main coastline
        land_poly = _build_land_polygon(
            main_coast, xmin, xmax, ymin, ymax,
            land_is_north=land_is_north,
        )
        if land_poly is not None:
            ax.fill(land_poly[:, 0], land_poly[:, 1],
                    color=color_land, transform=proj, zorder=1)

        # Fill closed segments (islands / small features)
        for seg in closed_segs:
            if len(seg) >= 3:
                ax.fill(seg[:, 0], seg[:, 1],
                        color=color_land, transform=proj, zorder=1)

        # Draw coastline on top of fill
        for seg in coast_segs:
            ax.plot(seg[:, 0], seg[:, 1],
                    color="#333333", linewidth=0.8, transform=proj, zorder=3)

    # ------------------------------------------------------------------
    # Control / extraction points
    # ------------------------------------------------------------------
    if points is not None:
        ax.scatter(
            points["x"].values, points["y"].values,
            color="#D32F2F", s=60, zorder=7,
            edgecolors="white", linewidths=0.7, transform=proj,
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor="#D32F2F",
                label=f"Points (n={len(points)})",
            )
        )
        if point_label_col and point_label_col in points.columns:
            for _, row in points.iterrows():
                ax.text(
                    row["x"] + point_label_offset,
                    row["y"] + point_label_offset,
                    str(row[point_label_col]),
                    transform=proj,
                    fontsize=7.5,
                    color="#B71C1C",
                    fontweight="bold",
                    zorder=7,
                )

    # ------------------------------------------------------------------
    # Lat/lon grid lines and manual labels
    # ------------------------------------------------------------------
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
        zorder=5,
    )
    from pyproj import Transformer as _Transformer
    _tr = _Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + utm_zone}",
                                 always_xy=True)
    for lon in np.arange(-10, 10, 1):
        xu, _ = _tr.transform(lon, (ymin + ymax) / 2 * 9e-6 + 36)
        # coarse lat estimate; accurate enough for tick placement
        xu, _ = _tr.transform(lon, 36.0)
        if xmin <= xu <= xmax:
            lbl = f"{abs(lon):.0f}\u00b0{'W' if lon < 0 else 'E'}"
            ax.text(xu, ymin, lbl, transform=proj, fontsize=7.5,
                    ha="center", va="top", color="#444444", zorder=6)
    for lat in np.arange(30, 45, 0.5):
        _, yu = _tr.transform(-3.0, lat)
        if ymin <= yu <= ymax:
            ax.text(xmin, yu, f"{lat:.1f}\u00b0N", transform=proj, fontsize=7.5,
                    ha="right", va="center", color="#444444", zorder=6)

    # ------------------------------------------------------------------
    # Scale bar
    # ------------------------------------------------------------------
    sb_len_m = round((xmax - xmin) / 6, -3)  # ~1/6 of map width, rounded to km
    sb_len_km = int(sb_len_m / 1000)
    bx = xmin + (xmax - xmin) * 0.82
    by = ymin + (ymax - ymin) * 0.05
    tick_h = (ymax - ymin) * 0.008
    ax.plot([bx, bx + sb_len_m], [by, by],
            color="black", lw=2, transform=proj, zorder=10,
            solid_capstyle="butt")
    ax.plot([bx, bx], [by - tick_h, by + tick_h],
            color="black", lw=1.5, transform=proj, zorder=10)
    ax.plot([bx + sb_len_m, bx + sb_len_m], [by - tick_h, by + tick_h],
            color="black", lw=1.5, transform=proj, zorder=10)
    ax.text(bx + sb_len_m / 2, by + tick_h * 1.5, f"{sb_len_km} km",
            ha="center", va="bottom", fontsize=8, transform=proj, zorder=10)

    # ------------------------------------------------------------------
    # Legend and title
    # ------------------------------------------------------------------
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.90,
        edgecolor="#AAAAAA",
        fancybox=False,
    )
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    show(fname)
    return ax


# ---------------------------------------------------------------------------
# Default colormaps and labels per DELFT3D-WAVE variable
# ---------------------------------------------------------------------------

# Blue → cyan → yellow → orange → red  (QuickPlot / DELFT3D style)
_CMAP_WAVE = LinearSegmentedColormap.from_list(
    "delft_wave",
    ["#0000CC", "#0055FF", "#00CCFF", "#FFFF00", "#FF8800", "#CC0000"],
    N=256,
)

_DELFT_FIELD_META = {
    # (colormap_key_or_object,  colorbar_label,  vmin_default, vmax_default)
    # None → auto-scale from data
    "hsign":    (_CMAP_WAVE,  "Hs (m)",              0,    None),
    "dir":      ("phase",     "Dir (\u00b0)",         0,    360),
    "pdir":     ("phase",     "Peak dir (\u00b0)",    0,    360),
    "period":   (_CMAP_WAVE,  "Tm (s)",               0,    15),
    "rtp":      (_CMAP_WAVE,  "Rtp (s)",              0,    15),
    "depth":    ("deep",      "Depth (m)",            0,    None),
    "veloc-x":  ("balance",   "Vel X (m/s)",         -1,    1),
    "veloc-y":  ("balance",   "Vel Y (m/s)",         -1,    1),
    "transp-x": ("balance",   "Transport X",         None,  None),
    "transp-y": ("balance",   "Transport Y",         None,  None),
    "dspr":     ("haline",    "Dir spread (\u00b0)",  0,    90),
    "dissip":   (_CMAP_WAVE,  "Dissipation",         None,  None),
    "leak":     (_CMAP_WAVE,  "Leak",                None,  None),
    "qb":       (_CMAP_WAVE,  "Qb",                   0,    1),
    "ubot":     (_CMAP_WAVE,  "Ubot (m/s)",           0,    None),
    "steepw":   (_CMAP_WAVE,  "Steepness",            0,    None),
    "wlength":  (_CMAP_WAVE,  "Wavelength (m)",       0,    None),
    "tps":      (_CMAP_WAVE,  "Tp (s)",               0,    15),
    "tm02":     (_CMAP_WAVE,  "Tm02 (s)",             0,    15),
    "tmm10":    (_CMAP_WAVE,  "Tm\u22121,0 (s)",      0,    15),
    "drtm01":   (_CMAP_WAVE,  "Drtm01 (s)",          None,  None),
    "setup":    ("balance",   "Setup (m)",           -0.5,   0.5),
    "fx":       ("balance",   "Fx",                  None,  None),
    "fy":       ("balance",   "Fy",                  None,  None),
    "windu":    ("balance",   "Wind U (m/s)",        -20,    20),
    "windv":    ("balance",   "Wind V (m/s)",        -20,    20),
}


def plot_wave_field(
    dat_path,
    grd_path,
    variable,
    coast_xyz=None,
    utm_zone=30,
    cmap=None,
    vmin=None,
    vmax=None,
    color_land="#EDE8DF",
    color_ocean="#C8DCF0",
    coast_gap_m=5000,
    land_is_north=True,
    margin=2000,
    quiver_variable=None,
    quiver_stride=2,
    quiver_scale=40,
    quiver_color="white",
    quiver_only=False,
    title=None,
    cbar_label=None,
    figsize=(10, 8),
    fname=None,
    depth_min=1.0,
):
    """Plot a DELFT3D-WAVE variable field on its curvilinear grid.

    Reads grid node coordinates from the ASCII ``.grd`` file and variable
    data from the NEFIS binary ``.dat`` file, then renders the field with
    ``pcolormesh`` on a cartopy UTM projection. An optional high-resolution
    coastline can be overlaid for geographic context.

    Args:
        dat_path (str | Path): Path to the NEFIS binary ``.dat`` output
            file for one model case.
        grd_path (str | Path): Path to the ASCII ``.grd`` grid file for
            the same case (used for node coordinates).
        variable (str): Wave variable to plot. Available names::

                hsign, dir, pdir, period, rtp, depth,
                veloc-x, veloc-y, transp-x, transp-y,
                dspr, dissip, leak, qb,
                ubot, steepw, wlength, tps, tm02, tmm10,
                drtm01, setup, fx, fy, windu, windv

        coast_xyz (str | Path | None): Path to a CSV with ``x`` and ``y``
            columns (same UTM CRS) for a high-resolution coastline.
            Defaults to None.
        utm_zone (int): UTM zone number for the cartopy projection.
            Defaults to 30.
        cmap (str | Colormap | None): Colormap override. If None, a
            sensible default is chosen per variable. Defaults to None.
        vmin (float | None): Colour scale minimum. Defaults to None
            (auto from data).
        vmax (float | None): Colour scale maximum. Defaults to None
            (auto from data).
        color_land (str): Fill colour for land. Defaults to ``"#EDE8DF"``.
        color_ocean (str): Background colour for ocean. Defaults to
            ``"#C8DCF0"``.
        coast_gap_m (float): Gap threshold in metres for splitting the
            coastline into segments. Defaults to 5000.
        land_is_north (bool): Close land polygon through the northern edge
            of the extent. Defaults to True.
        margin (float): Extra margin in metres around the grid extent.
            Defaults to 2000.
        quiver_variable (str | None): Variable to use for direction arrows.
            Typically ``"dir"`` (mean wave direction in degrees, measured
            clockwise from North, direction of provenance: 0° = from North
            → arrow points South). When set, arrows are drawn with length
            and colour proportional to the main *variable* magnitude using
            the same colormap. Defaults to None (no arrows).
        quiver_stride (int): Subsample every *n* grid cells in both
            directions before drawing arrows. Defaults to 12.
        quiver_scale (float): Quiver scale passed to ``ax.quiver``; smaller
            values produce longer arrows. With magnitude-scaled arrows a
            value around 30–50 works well. Defaults to 40.
        quiver_color (str): Ignored when *quiver_variable* is set (arrows
            are coloured by magnitude). Kept for API compatibility.
        title (str | None): Figure title. If None, auto-generated from
            *variable* and the case directory name. Defaults to None.
        cbar_label (str | None): Colorbar label override. Defaults to None
            (uses variable metadata).
        figsize (tuple): Figure size in inches. Defaults to (10, 8).
        fname (str | None): Output file path. If None, displays
            interactively. Defaults to None.
        depth_min (float): Minimum depth (m) for cells to be included in the
            plot. Cells shallower than this threshold are masked to NaN and
            shown as ocean background. Useful for hiding the very shallow
            coastal fringe (depth < 1 m) where DELFT3D outputs Hs ≈ 0 due
            to complete wave breaking, which otherwise creates a dark-blue
            band along the coast. Set to 0 or None to disable. Not applied
            when *variable* is ``"depth"``. Defaults to 1.0.

    Returns:
        matplotlib.axes.GeoAxes: The cartopy axes with the field plot.

    Examples:
        >>> from pathlib import Path
        >>> from environmentaltools.models import delft3d
        >>>
        >>> delft3d.plot_wave_field(
        ...     dat_path=Path("caso_001/wavm-guad-Alboran_int.dat"),
        ...     grd_path=Path("caso_001/Alboran_int.grd"),
        ...     variable="hsign",
        ...     coast_xyz=Path("linea_costa_oficial_utm30n.xyz"),
        ...     vmax=4.0,
        ...     title="Hs \u2014 caso 001",
        ...     fname="hs_caso001.png",
        ... )
    """
    ccrs = _cartopy_crs()
    from environmentaltools.graphics.utils import show

    dat_path = Path(dat_path)
    grd_path = Path(grd_path)

    # ------------------------------------------------------------------
    # Grid node coordinates from .grd
    # ------------------------------------------------------------------
    X, Y, _ = read_delft_grd(grd_path)

    # ------------------------------------------------------------------
    # Variable field from .dat
    # ------------------------------------------------------------------
    gp  = _delft_grid_params(dat_path.parent, dat_path.name, grd_path.name)
    val = _delft_read_var(dat_path, variable, gp, fill_threshold=_NEFIS_FILL)

    # Mask inactive cells using the XP/YP coordinate mask from the .dat file.
    # Active sea cells have valid UTM coordinates; land/outside cells have XP=0.
    _, _, _active = _delft_read_coords(dat_path, gp, fill_threshold=_NEFIS_FILL)
    val = np.where(~_active, np.nan, val)

    # Also mask cells where the .grd defines no grid node (outside model extent).
    val = np.where(np.isnan(X) | np.isnan(Y), np.nan, val)

    # Mask very shallow coastal cells (depth < depth_min) as NaN so they
    # appear as ocean background rather than dark-blue Hs≈0.  These cells
    # are correctly computed by DELFT3D but show near-zero wave energy due
    # to complete breaking in the intertidal fringe — hiding them avoids a
    # misleading dark band along the coast.  Skip when plotting depth itself.
    if depth_min and variable != "depth":
        _dep = _delft_read_var(dat_path, "depth", gp, fill_threshold=_NEFIS_FILL)
        val = np.where(_dep < depth_min, np.nan, val)

    if variable in ("dir", "pdir"):
        val = (270.0 - val) % 360.0

    n_valid = int(np.sum(~np.isnan(val)))
    print(f"[plot_delft] '{variable}'  shape={val.shape}  "
          f"valid={n_valid}/{val.size}  "
          f"min={np.nanmin(val) if n_valid else float('nan'):.4f}  "
          f"max={np.nanmax(val) if n_valid else float('nan'):.4f}")

    # ------------------------------------------------------------------
    # Colormap and label
    # ------------------------------------------------------------------
    meta_cmap, meta_label, meta_vmin, meta_vmax = _DELFT_FIELD_META.get(
        variable, (_CMAP_WAVE, variable, None, None)
    )
    if cmap is None:
        if isinstance(meta_cmap, Colormap):
            cmap = meta_cmap          # already a Colormap object (e.g. _CMAP_WAVE)
        else:
            try:
                cmap = getattr(cmocean.cm, meta_cmap)
            except AttributeError:
                cmap = meta_cmap
    # Resolve string colormaps (from caller or fallback) to Colormap objects.
    if isinstance(cmap, str):
        try:
            cmap = getattr(cmocean.cm, cmap)
        except AttributeError:
            cmap = mpl.colormaps[cmap]
    # Make masked/NaN cells appear as ocean background instead of the
    # colormap's default "bad" colour (often white).
    cmap = copy.copy(cmap)
    cmap.set_bad(color=color_ocean)
    if cbar_label is None:
        cbar_label = meta_label
    # Apply per-variable default scale when caller did not specify vmin/vmax
    if vmin is None:
        vmin = meta_vmin
    if vmax is None:
        vmax = meta_vmax


    # plt.pcolormesh(X,Y,np.ma.masked_invalid(val))
    # plt.show()

    # ------------------------------------------------------------------
    # Figure and projection
    # ------------------------------------------------------------------
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    ax.set_facecolor(color_ocean)

    xmin = float(np.nanmin(X)) - margin
    xmax = float(np.nanmax(X)) + margin
    ymin = float(np.nanmin(Y)) - margin
    ymax = float(np.nanmax(Y)) + margin
    # set_extent with a projected CRS can fail silently in some Cartopy
    # versions.  Use set_xlim/set_ylim directly in the axes (projected)
    # coordinate space instead — these always work.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ------------------------------------------------------------------
    # Variable field (pcolormesh on curvilinear grid)  [zorder=2]
    # Build a shared Normalize so pcolormesh and quiver use identical scaling.
    # ------------------------------------------------------------------
    _vmin = vmin if vmin is not None else float(np.nanmin(val))
    _vmax = vmax if vmax is not None else float(np.nanmax(val))
    norm  = Normalize(vmin=_vmin, vmax=_vmax)

    # quiver_only without a quiver_variable is a no-op — show the field anyway.
    if quiver_only and quiver_variable is None:
        quiver_only = False


    # plt.imshow(val)
    # plt.show()


    if not quiver_only:
        pm = ax.pcolormesh(
            X, Y, np.ma.masked_invalid(val),
            cmap=cmap, norm=norm,
            transform=proj, zorder=2, shading="auto",
        )
        cbar = fig.colorbar(pm, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.set_ylabel(cbar_label, fontsize=9)

    # ------------------------------------------------------------------
    # Coastline geometry — computed first so it can be used as a mask
    # for the quiver arrows before any drawing takes place.
    # ------------------------------------------------------------------
    land_poly   = None   # numpy array (N,2) of the land polygon vertices
    coast_segs  = []
    closed_segs = []

    if coast_xyz is not None:
        coast_segs = _load_coast_segments(coast_xyz, gap_m=coast_gap_m)
        open_idx   = [i for i, s in enumerate(coast_segs)
                      if np.hypot(s[-1, 0] - s[0, 0], s[-1, 1] - s[0, 1]) > coast_gap_m]
        closed_idx = [i for i in range(len(coast_segs)) if i not in open_idx]
        open_segs   = [coast_segs[i] for i in open_idx]
        closed_segs = [coast_segs[i] for i in closed_idx]

        if len(open_segs) >= 2:
            s0, s1 = open_segs[0], open_segs[1]
            if np.allclose(s0[-1], s1[0], atol=10):
                main_coast = np.vstack([s0, s1[1:]])
            elif np.allclose(s1[-1], s0[0], atol=10):
                main_coast = np.vstack([s1, s0[1:]])
            else:
                main_coast = np.vstack(open_segs)
        elif len(open_segs) == 1:
            main_coast = open_segs[0]
        else:
            main_coast = np.vstack(coast_segs)

        land_poly = _build_land_polygon(
            main_coast, xmin, xmax, ymin, ymax, land_is_north=land_is_north,
        )

    # ------------------------------------------------------------------
    # Direction quiver (optional)  [zorder=3]
    # Arrows are:
    #   • coloured by the main variable magnitude (same cmap/norm as field)
    #   • sized proportional to magnitude (longer = higher value)
    #   • geometrically clipped to ocean (land polygon mask)
    # ------------------------------------------------------------------
    if quiver_variable is not None:
        dir_arr = _delft_read_var(dat_path, quiver_variable, gp, fill_threshold=_NEFIS_FILL)
        # Mask dir_arr with the exact same masks applied to val.
        dir_arr = np.where(~_active, np.nan, dir_arr)
        dir_arr = np.where(np.isnan(X) | np.isnan(Y), np.nan, dir_arr)
        if depth_min and variable != "depth":
            dir_arr = np.where(_dep < depth_min, np.nan, dir_arr)

        # Subsample — same X, Y, val as pcolormesh.
        st   = quiver_stride
        Xs   = X[::st, ::st]
        Ys   = Y[::st, ::st]
        Ds   = dir_arr[::st, ::st]
        Mags = val[::st, ::st]

        # Aplicar la misma transformación que usa pcolormesh para dir/pdir,
        # y luego fórmula cartesiana estándar U=cos, V=sin.
        # (270 - dir) % 360 convierte náutico-FROM a cartesiano-TO (CCW desde Este):
        #   dir=0   (del N, va al S) → 270° → (cos270, sin270) = (0,-1) Sur  ✓
        #   dir=90  (del E, va al O) → 180° → (cos180, sin180) = (-1,0) Oeste ✓
        #   dir=270 (del O, va al E) →   0° → (cos0,   sin0)   = (1, 0) Este  ✓
        if quiver_variable in ("dir", "pdir"):
            Ds = (270.0 - Ds) % 360.0
        rad = np.deg2rad(Ds)
        Us  = (-1) * np.sin(rad)
        Vs  = (-1) * np.cos(rad)

        mask_q = ~(np.isnan(Xs) | np.isnan(Ys) | np.isnan(Ds) | np.isnan(Mags))
        # Land polygon clipping removed — land fill (zorder=5) draws on top
        # and naturally hides any arrows that fall on land.

        # Scale arrow length proportional to magnitude.
        # Floor at 0.25 so arrows remain visible even in near-calm cells;
        # colour each arrow with the same cmap/norm as pcolormesh (QuickPlot style).
        mag_vals = Mags[mask_q]
        if _vmax > 0:
            scale_f = 0.25 + 0.75 * np.clip(mag_vals / _vmax, 0.0, 1.0)
        else:
            scale_f = np.ones_like(mag_vals)
        _quiver_kw = dict(
            transform=proj,
            scale=quiver_scale,
            scale_units="width",
            width=0.001,
            headwidth=3,
            headlength=4,
            linewidth=0,
        )
        if quiver_only:
            # quiver_only: flechas coloreadas por magnitud + colorbar propio.
            ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                color="white", alpha=1.0, zorder=3, **_quiver_kw,
            )
            qv = ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                mag_vals, cmap=cmap, norm=norm,
                alpha=0.9, zorder=4, **_quiver_kw,
            )
            import matplotlib.cm as _mcm
            sm = _mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
            cbar.ax.set_ylabel(cbar_label, fontsize=9)
        else:
            # Con pcolormesh de fondo: flechas en color fijo (quiver_color).
            ax.quiver(
                Xs[mask_q], Ys[mask_q],
                Us[mask_q] * scale_f, Vs[mask_q] * scale_f,
                color=quiver_color, alpha=0.9, zorder=3, **_quiver_kw,
            )

    # ------------------------------------------------------------------
    # Coastline and land fill  [zorder=5/6] — above quiver (zorder 3/4)
    # ------------------------------------------------------------------
    if coast_xyz is not None:
        if land_poly is not None:
            ax.fill(land_poly[:, 0], land_poly[:, 1],
                    color=color_land, transform=proj, zorder=5)
        for seg in closed_segs:
            if len(seg) >= 3:
                ax.fill(seg[:, 0], seg[:, 1],
                        color=color_land, transform=proj, zorder=5)
        for seg in coast_segs:
            ax.plot(seg[:, 0], seg[:, 1],
                    color="#333333", linewidth=0.8, transform=proj, zorder=6)

    # ------------------------------------------------------------------
    # Lat/lon gridlines with manual labels
    # ------------------------------------------------------------------
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.4, color="gray", alpha=0.5, linestyle="--", zorder=6,
    )
    from pyproj import Transformer as _Tr
    _tr = _Tr.from_crs("EPSG:4326", f"EPSG:{32600 + utm_zone}", always_xy=True)
    for lon in np.arange(-10, 10, 1):
        xu, _ = _tr.transform(lon, 36.0)
        if xmin <= xu <= xmax:
            lbl = f"{abs(lon):.0f}\u00b0{'W' if lon < 0 else 'E'}"
            ax.text(xu, ymin, lbl, transform=proj, fontsize=7,
                    ha="center", va="top", color="#444444", zorder=7)
    for lat in np.arange(30, 45, 0.5):
        _, yu = _tr.transform(-3.0, lat)
        if ymin <= yu <= ymax:
            ax.text(xmin, yu, f"{lat:.1f}\u00b0N", transform=proj, fontsize=7,
                    ha="right", va="center", color="#444444", zorder=7)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    if title is None:
        title = f"{variable.upper()} \u2014 {dat_path.parent.name}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    show(fname)
    return ax
