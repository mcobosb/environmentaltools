"""Geometry of a BARRIER channel from a DEM (raster or xyz points) and a channel axis.

Builds the two geometry files of a case, ``cross-sections.csv`` and ``along_channel_data.csv``, from a DEM and an
axis (:func:`build_geometry`) or for an idealised rectangular or trapezoidal channel (:func:`prismatic_geometry`)::

    cs, along, report = barrier.build_geometry("dem.tif", "axis.shp", dx=200, half_width=2000, top_level=5,
                                               out_dir="base_case")

Steps (each one is also a function):

1. :func:`section_lines`: sections every ``dx`` along the axis (given by the user, from the upstream boundary to
   the mouth), normal to it. The angles follow BARRIER: the left bank (looking downstream) is at
   ``xutm + |xl| cos(AngMi)``, the right bank at ``xutm + |xr| cos(AngMd)`` (radians from the east).
2. :func:`sample_profiles`: the DEM along each section line.
3. :func:`section_table`: the table of each section, one row per water level. At each level the wet part is the
   continuous stretch below the level that contains the bed (a dip behind higher ground does not count until the
   level tops it); its ends are the banks ``xl`` and ``xr`` (m from the axis, left < 0), ``B = xr − xl``.
4. :func:`smooth_sections` (optional): tables averaged with their neighbours along the channel. Sections taken one
   by one from a DEM change too much from one to the next, and BARRIER gives a saw-tooth mean level.

Method of the BARRIER geometry of the Guadalquivir (2026, tasks 04 and 06 of PRO-2025_P_BARRIER_GUADALQUIVIR).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import CROSS_SECTION_COLUMNS, check_cross_sections, write_csv
from .read import distance_to_mouth

#: Columns of ``along_channel_data.csv`` in the order BARRIER reads them (by position, not by name).
ALONG_COLUMNS = ["x", "z", "nmann", "xutm", "yutm", "AngMd", "AngMi", "beta", "Sf", "kh"]
#: Suffixes read as xyz points (anything else is opened as a raster).
XYZ_SUFFIXES = (".xyz", ".csv", ".txt", ".dat")


def _wrap(angle):
    """Angle in (−π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def read_axis(axis):
    """Channel axis as an ``(n, 2)`` array of points, from the upstream boundary to the mouth.

    Args:
        axis (str | Path | shapely.LineString | array_like): Line file (shapefile, GeoJSON, GeoPackage…: its
            line, or the merged lines), csv/txt with the points (columns ``x``, ``y``, else the first two), a
            LineString or an array of points.

    Returns:
        np.ndarray: Points without consecutive duplicates.
    """
    if isinstance(axis, (str, Path)):
        f = Path(axis)
        if f.suffix.lower() in XYZ_SUFFIXES:
            df = pd.read_csv(f, sep=None, engine="python")
            cols = ["x", "y"] if {"x", "y"} <= set(df.columns) else list(df.columns[:2])
            pts = df[cols].to_numpy(dtype=float)
        else:
            import geopandas as gpd
            from shapely.ops import linemerge

            geom = gpd.read_file(f).geometry
            line = geom.iloc[0] if len(geom) == 1 else linemerge(list(geom))
            if line.geom_type != "LineString":
                raise ValueError(f"{f.name}: the axis must be one line (got {line.geom_type})")
            pts = np.asarray(line.coords)[:, :2]
    elif hasattr(axis, "coords"):
        pts = np.asarray(axis.coords)[:, :2]
    else:
        pts = np.asarray(axis, dtype=float)[:, :2]
    keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("The axis needs at least two different points")
    return pts


def section_lines(axis, dx, smooth=None):
    """Sections every ``dx`` along the axis, normal to it.

    The axis length is divided in equal steps as close to ``dx`` as possible, so the first section is the start of
    the axis (upstream boundary, ``x = 0``) and the last one its end (the mouth).

    Args:
        axis: Channel axis (see :func:`read_axis`), from the upstream boundary to the mouth.
        dx (float): Distance between sections along the axis (m).
        smooth (float | None): Standard deviation (m along the axis) of the Gaussian smoothing of the axis used for
            the directions of the sections; larger values keep the sections of a bend from crossing.

    Returns:
        pd.DataFrame: One row per section: ``x`` (m from the upstream boundary), ``xutm``, ``yutm``, ``AngMd``,
        ``AngMi`` (radians).
    """
    pts = read_axis(axis)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    n = max(1, int(round(s[-1] / dx)))
    x = np.linspace(0.0, s[-1], n + 1)
    xu, yu = np.interp(x, s, pts[:, 0]), np.interp(x, s, pts[:, 1])
    # directions from the axis itself, densely resampled (and smoothed), not from the sections: a bend between two
    # sections still turns them
    step = min(dx, s[-1]) / 10
    xs = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step)) + 1)
    px, py = np.interp(xs, s, pts[:, 0]), np.interp(xs, s, pts[:, 1])
    if smooth:
        from scipy.ndimage import gaussian_filter1d

        sig = smooth / (xs[1] - xs[0])
        px, py = gaussian_filter1d(px, sig, mode="nearest"), gaussian_filter1d(py, sig, mode="nearest")
    theta = np.arctan2(np.gradient(py), np.gradient(px))
    theta = np.interp(x, xs, np.unwrap(theta))
    return pd.DataFrame({"x": x, "xutm": xu, "yutm": yu, "AngMd": _wrap(theta - np.pi / 2),
                         "AngMi": _wrap(theta + np.pi / 2)})


def section_points(sections, offsets):
    """UTM coordinates of points of each section line at ``offsets`` (m from the axis, left < 0).

    Args:
        sections (pd.DataFrame): Columns ``xutm``, ``yutm``, ``AngMd``, ``AngMi`` (see :func:`section_lines`).
        offsets (array_like): One offset for all sections, an array of offsets for all sections, or one row of
            offsets per section.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``x`` and ``y`` with the shape of the offsets per section.
    """
    off = np.asarray(offsets, dtype=float)
    n = len(sections)
    off = np.broadcast_to(off if off.ndim == 2 else off[None, :] if off.ndim == 1 else off.reshape(1, 1),
                          (n, off.shape[-1] if off.ndim else 1))
    col = lambda c: sections[c].to_numpy(dtype=float)[:, None]   # noqa: E731
    ang = np.where(off < 0, col("AngMi"), col("AngMd"))
    return col("xutm") + np.abs(off) * np.cos(ang), col("yutm") + np.abs(off) * np.sin(ang)


def _is_xyz(dem):
    return isinstance(dem, pd.DataFrame) or (isinstance(dem, (str, Path)) and Path(dem).suffix.lower() in XYZ_SUFFIXES)


def read_xyz(file_name):
    """Points ``x``, ``y``, ``z`` of an xyz file (separated by spaces, tabs, commas or semicolons; with or without a
    header line; columns ``x``, ``y``, ``z`` if named, else the first three)."""
    f = Path(file_name)
    with open(f) as fh:
        first = fh.readline()
    sep = "," if "," in first else ";" if ";" in first else r"\s+"
    has_header = any(c.isalpha() for c in first.replace("e", "").replace("E", ""))
    df = pd.read_csv(f, sep=sep, header=0 if has_header else None, comment="#")
    if has_header and {"x", "y", "z"} <= set(df.columns.str.strip().str.lower()):
        df.columns = df.columns.str.strip().str.lower()
        return df[["x", "y", "z"]].astype(float)
    df = df.iloc[:, :3].astype(float)
    df.columns = ["x", "y", "z"]
    return df


def sample_profiles(dem, sections, half_width, step=None):
    """DEM along each section line, every ``step`` from ``−half_width`` (left bank side) to ``half_width``.

    Args:
        dem (str | Path | pd.DataFrame): Raster file (GeoTIFF, VRT, ASCII grid…: the nearest cell, its nodata as NaN)
            or xyz points (a file with one of :data:`XYZ_SUFFIXES`, see :func:`read_xyz`, or a DataFrame with ``x``,
            ``y``, ``z``: linear interpolation on their triangulation, NaN outside). Same coordinates as the axis.
        sections (pd.DataFrame): Section lines (:func:`section_lines`).
        half_width (float): Length of the profile on each side of the axis (m); it must reach beyond the banks at
            the highest level of the tables.
        step (float | None): Distance between profile points (m); ``None``: the cell size of a raster, 1 m for xyz.

    Returns:
        tuple[np.ndarray, np.ndarray]: Offsets ``s`` (m from the axis, left < 0) and elevations, one row per
        section.
    """
    if _is_xyz(dem):
        from scipy.interpolate import LinearNDInterpolator

        pts = dem if isinstance(dem, pd.DataFrame) else read_xyz(dem)
        step = step or 1.0
        s = np.arange(-half_width, half_width + step / 2, step)
        px, py = section_points(sections, s)
        interp = LinearNDInterpolator(pts[["x", "y"]].to_numpy(dtype=float), pts["z"].to_numpy(dtype=float))
        return s, interp(px, py)

    import rasterio

    with rasterio.open(dem) as src:
        step = step or float(min(abs(src.res[0]), abs(src.res[1])))
        s = np.arange(-half_width, half_width + step / 2, step)
        px, py = section_points(sections, s)
        z = np.array([v[0] for v in src.sample(zip(px.ravel(), py.ravel()))], dtype=float).reshape(px.shape)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
    z[~np.isfinite(z)] = np.nan
    return s, z


def section_table(s, z, x=0.0, level_step=0.05, top_level=None, bed_half_width=None):
    """Table of one section for BARRIER from its profile: one row per water level, from the bed to ``top_level``.

    At each level ``L`` the wet part is the continuous stretch of the profile below ``L`` that contains the bed:
    its banks are the first points at or above ``L`` on each side (the running maximum from the bed outwards), so a
    dip behind higher ground does not count until ``L`` tops it, and the banks open and ``A`` and ``B`` grow with
    the level. Bank positions are interpolated linearly; a point without data (NaN) is a wall.

    Args:
        s (array_like): Offsets of the profile points (m from the axis, left < 0).
        z (array_like): Elevations of the profile.
        x (float): Position of the section (m from the upstream boundary), written in the table.
        level_step (float): Vertical step of the table (m).
        top_level (float): Highest level of the table (absolute, same datum as ``z``).
        bed_half_width (float | None): The bed (lowest point) is searched for within ``±bed_half_width`` of the
            axis; ``None``: the whole profile.

    Returns:
        pd.DataFrame: Columns :data:`~environmentaltools.models.barrier.CROSS_SECTION_COLUMNS` (``eta`` = water depth
        above the bed, ``sigma = B``, ``beta = 1``) and ``open_end`` (True when the wet part reaches an end of the
        profile or a point without data).
    """
    s, zz = np.asarray(s, dtype=float), np.asarray(z, dtype=float)
    zz = np.where(np.isnan(zz), np.inf, zz)
    inner = np.isfinite(zz) & (np.abs(s) <= (np.inf if bed_half_width is None else bed_half_width))
    if not inner.any():
        raise ValueError(f"Section x = {x:g}: no DEM data near the axis")
    imin = int(np.flatnonzero(inner)[np.argmin(zz[inner])])
    zmin = zz[imin]
    if top_level is None or top_level <= zmin:
        raise ValueError(f"Section x = {x:g}: top_level must be above its bed ({zmin:g})")
    left_max = np.maximum.accumulate(zz[imin::-1])
    right_max = np.maximum.accumulate(zz[imin:])
    rows = [dict(x=x, z=zmin, eta=0.0, B=0.0, A=0.0, P=0.0, Rh=0.0, sigma=0.0, xl=s[imin], xr=s[imin], beta=1,
                 open_end=False)]
    for L in np.arange(zmin + level_step, top_level + level_step / 2, level_step):
        il = imin - np.searchsorted(left_max, L, side="left")      # first dry point on each side (may be outside)
        ir = imin + np.searchsorted(right_max, L, side="left")
        open_end = il < 0 or ir >= len(s) or not np.isfinite(zz[il]) or not np.isfinite(zz[ir])
        lo, hi = max(il + 1, 0), min(ir - 1, len(s) - 1)
        ws, wz = s[lo:hi + 1], zz[lo:hi + 1]
        xl = s[il] + (L - zz[il]) * (s[lo] - s[il]) / (zz[lo] - zz[il]) if il >= 0 and np.isfinite(zz[il]) else s[lo]
        xr = s[hi] + (L - zz[hi]) * (s[ir] - s[hi]) / (zz[ir] - zz[hi]) if ir < len(s) and np.isfinite(zz[ir]) \
            else s[hi]
        px, pz = np.r_[xl, ws, xr], np.r_[L, wz, L]
        A = np.trapezoid(L - pz, px)
        P = np.sum(np.hypot(np.diff(px), np.diff(pz)))
        rows.append(dict(x=x, z=zmin, eta=L - zmin, B=xr - xl, A=A, P=P, Rh=A / P if P > 0 else 0.0, sigma=xr - xl,
                         xl=xl, xr=xr, beta=1, open_end=open_end))
    return pd.DataFrame(rows)


def smooth_sections(cs, sigma, level_step=0.05):
    """Tables of the sections averaged with their neighbours along the channel (Gaussian filter in distance).

    Each table is taken to a common grid of absolute levels (``A = B = P = 0`` below the bed of the section) and
    ``A(L)``, ``B(L)``, ``P(L)`` and the centre of the banks are averaged with weights ``exp(−½ (Δx / sigma)²)``, cut
    at ±3 sigma. As a mean of growing functions grows, the tables stay monotonic and ``B`` stays ``dA/dL``. The new
    bed is the weighted mean of the beds; the area of deeper neighbours below it is discounted. The banks are the
    smoothed centre ± ``B / 2``.

    Args:
        cs (pd.DataFrame): Cross-section tables (:data:`~environmentaltools.models.barrier.CROSS_SECTION_COLUMNS`).
        sigma (float): Standard deviation of the filter (m along the channel); 0 or None returns the tables as they
            are.
        level_step (float): Vertical step of the new tables (m).

    Returns:
        pd.DataFrame: Smoothed tables (:data:`~environmentaltools.models.barrier.CROSS_SECTION_COLUMNS`).
    """
    cs = cs[CROSS_SECTION_COLUMNS]
    if not sigma:
        return cs.reset_index(drop=True)
    zmin = cs.groupby("x")["z"].first().min()
    top = (cs["z"] + cs["eta"]).max()
    grid = np.arange(np.floor(zmin / level_step) * level_step, top + level_step / 2, level_step)
    xs = np.sort(cs["x"].unique())
    tab = {k: np.zeros((len(xs), len(grid))) for k in ("A", "B", "P", "c")}
    zraw = np.zeros(len(xs))
    for i, (_, sec) in enumerate(cs.groupby("x", sort=True)):
        sec = sec.sort_values("eta")
        lev = (sec["z"] + sec["eta"]).to_numpy()
        zraw[i] = lev[0]
        for k in ("A", "B", "P"):
            tab[k][i] = np.interp(grid, lev, sec[k].to_numpy(), left=0.0)
        tab["c"][i] = np.interp(grid, lev, ((sec["xl"] + sec["xr"]) / 2).to_numpy())
    d = (xs[:, None] - xs[None, :]) / sigma
    w = np.where(np.abs(d) <= 3, np.exp(-0.5 * d ** 2), 0.0)
    w /= w.sum(axis=1, keepdims=True)
    A, B, P, C = (w @ tab[k] for k in ("A", "B", "P", "c"))
    znew = w @ zraw
    rows = []
    for i, x in enumerate(xs):
        zb = znew[i]
        below = float(np.interp(zb, grid, A[i]))                  # area of deeper neighbours under the new bed
        first = np.ceil(np.round(zb / level_step, 6)) * level_step + level_step   # round: -4.9999999 is -5
        lev = np.r_[zb, np.arange(first, grid[-1] + level_step / 2, level_step)]
        a = np.maximum(np.interp(lev, grid, A[i]) - below, 0.0)
        b, p, c = (np.interp(lev, grid, M[i]) for M in (B, P, C))
        a[0] = b[0] = p[0] = 0.0                                  # bed row
        rows.append(pd.DataFrame(dict(x=x, z=zb, eta=lev - zb, B=b, A=a, P=p,
                                      Rh=np.divide(a, p, out=np.zeros_like(a), where=p > 0), sigma=b,
                                      xl=c - b / 2, xr=c + b / 2, beta=1)))
    return pd.concat(rows, ignore_index=True)


def copy_downstream(cs, n):
    """The last ``n`` sections replaced by copies of the one before them (flat bed at the mouth).

    A bed slope in the last sections can make BARRIER unstable at the tidal boundary (in the Guadalquivir it blew up
    at the mouth after ~4 h; with one copy it was stable).
    """
    if not n:
        return cs
    xs = np.sort(cs["x"].unique())
    src = cs[cs["x"] == xs[-n - 1]]
    return pd.concat([cs[~cs["x"].isin(xs[-n:])], *(src.assign(x=x) for x in xs[-n:])], ignore_index=True)


def _value_at(sec, level, col):
    """Value of ``col`` of a section table at an absolute level (0 below the bed)."""
    lev = (sec["z"] + sec["eta"]).to_numpy()
    return float(np.interp(level, lev, sec[col].to_numpy(), left=0.0))


def crossing_sections(sections, cs, level=None):
    """Sections whose wet line at ``level`` crosses the next one (in a bend, sections too long or too close).

    Args:
        sections (pd.DataFrame): Section lines (:func:`section_lines`).
        cs (pd.DataFrame): Their tables.
        level (float | None): Absolute level; ``None``: the top of the tables.

    Returns:
        list[float]: ``x`` of the sections that cross the next one.
    """
    from shapely.geometry import LineString

    tables = {x: t for x, t in cs.groupby("x", sort=True)}
    lines = []
    for _, sec in sections.iterrows():
        t = tables[sec["x"]]
        lev = (t["z"] + t["eta"]).max() if level is None else level
        xl, xr = _value_at(t, lev, "xl"), _value_at(t, lev, "xr")
        if xr - xl <= 0:
            lines.append(None)
            continue
        px, py = section_points(sec.to_frame().T, [xl, 0.0, xr])
        lines.append(LineString(np.c_[px[0], py[0]]))
    return [float(sections["x"].iloc[i]) for i in range(len(lines) - 1)
            if lines[i] is not None and lines[i + 1] is not None and lines[i].intersects(lines[i + 1])]


def _finish(cs, sections, top_level, check_level, nmann, kh, out_dir, verbose, open_end=None, bed_offset=None):
    """Along-channel table, report, warnings and files of a geometry (shared by :func:`build_geometry` and
    :func:`prismatic_geometry`)."""
    open_end = pd.Series(False, index=sections["x"]) if open_end is None else open_end
    bed_offset = pd.Series(0.0, index=sections["x"]) if bed_offset is None else bed_offset
    beds = cs.groupby("x", sort=True)["z"].first()
    along = pd.DataFrame({"x": sections["x"], "z": sections["x"].map(beds), "nmann": float(nmann),
                          "xutm": sections["xutm"], "yutm": sections["yutm"], "AngMd": sections["AngMd"],
                          "AngMi": sections["AngMi"], "beta": 1.0, "Sf": 1.0})
    if kh is not None:
        along["kh"] = float(kh)

    level = top_level if check_level is None else check_level
    tables = dict(tuple(cs.groupby("x", sort=True)))
    area = np.array([_value_at(tables[x], level, "A") for x in sections["x"]])
    report = pd.DataFrame({
        "x": sections["x"], "km": distance_to_mouth(sections["x"]).round(3), "bed": sections["x"].map(beds),
        "bed_offset": sections["x"].map(bed_offset),
        "width": [_value_at(tables[x], level, "B") for x in sections["x"]], "area": area,
        "area_jump": np.r_[np.nan, np.abs(np.diff(np.log(np.where(area > 0, area, np.nan))))],
        "open_end": sections["x"].map(open_end).to_numpy(dtype=bool)})
    crossing = set(crossing_sections(sections, cs, level))
    report["crosses_next"] = report["x"].isin(crossing)

    bad = check_cross_sections(cs)
    warnings = []
    if len(bad):
        warnings.append(f"{bad['x'].nunique()} sections with non-monotonic tables (see check_cross_sections)")
    if report["open_end"].any():
        warnings.append(f"{int(report['open_end'].sum())} sections where the water reaches the end of the profile "
                        f"or a gap of the DEM at {top_level:g} m (give a larger half_width, or check the DEM)")
    if crossing:
        warnings.append(f"{len(crossing)} sections cross the next one at {level:g} m (larger axis_smooth or dx): "
                        f"x = {', '.join(f'{x:g}' for x in sorted(crossing)[:10])}")
    high = report[report["bed"] >= 0]
    if len(high):
        warnings.append(f"{len(high)} sections with the bed at 0 m or above: BARRIER starts with the water level at "
                        "0 (initial_conditions: calm), so they start dry")
    for w in warnings:
        print(f"Warning: {w}")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(cs, out_dir / "cross-sections.csv")
        write_csv(along, out_dir / "along_channel_data.csv")
    if verbose:
        print(f"{len(sections)} sections every {np.diff(sections['x']).mean():.1f} m along "
              f"{sections['x'].iloc[-1] / 1000:.2f} km; bed from {report['bed'].min():.2f} to "
              f"{report['bed'].max():.2f} m; width at {level:g} m from {report['width'].min():.0f} to "
              f"{report['width'].max():.0f} m" + (f"; written in {out_dir}" if out_dir is not None else ""))
    return cs, along, report


def build_geometry(dem, axis=None, dx=None, half_width=None, top_level=None, sections=None, profile_step=None,
                   level_step=0.05, bed_half_width=None, axis_smooth=None, smooth=None, downstream_copies=0,
                   nmann=0.03, kh=None, check_level=None, out_dir=None, verbose=True):
    """Geometry of a BARRIER channel (``cross-sections.csv`` and ``along_channel_data.csv``) from a DEM and an axis.

    Args:
        dem (str | Path | pd.DataFrame): Raster or xyz points (see :func:`sample_profiles`), in the same horizontal
            coordinates as the axis and the same vertical datum as the tide at the mouth.
        axis: Channel axis from the upstream boundary to the mouth (see :func:`read_axis`). It must run inside the
            channel: BARRIER places the banks at ``|xl|`` and ``|xr|`` from it.
        dx (float): Distance between sections along the axis (m).
        half_width (float): Length of each profile on each side of the axis (m).
        top_level (float): Highest level of the tables (absolute, m): above the highest water level expected.
        sections (pd.DataFrame | None): Section lines given instead of ``axis`` and ``dx`` (columns ``x``, ``xutm``,
            ``yutm``, ``AngMd``, ``AngMi``), e.g. those of an existing case.
        profile_step (float | None): Distance between profile points (m; see :func:`sample_profiles`).
        level_step (float): Vertical step of the tables (m).
        bed_half_width (float | array_like | None): The bed of each section is searched for within this distance of
            the axis (one value, or one per section); ``None``: the whole profile.
        axis_smooth (float | None): Smoothing of the axis for the directions of the sections (m; see
            :func:`section_lines`).
        smooth (float | None): Smoothing of the tables along the channel (sigma in m; see :func:`smooth_sections`).
            500 m worked for the Guadalquivir (200 m sections); ``None``: tables as taken from the DEM.
        downstream_copies (int): The last sections are copies of the one before them (see :func:`copy_downstream`).
        nmann (float): Manning n of every section (s/m^(1/3)), to be calibrated.
        kh (float | None): Longitudinal dispersion of every section (m²/s); ``None`` leaves the column out (BARRIER
            then uses ``transport.salinity.dispersion_kh``).
        check_level (float | None): Level of the report (width, area, area jump) and of the check of crossing
            sections, e.g. the highest tide; ``None``: ``top_level``.
        out_dir (str | Path | None): Case folder where both files are written (every digit, see
            :func:`~environmentaltools.models.barrier.write_csv`).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ``cross_sections`` and ``along_channel`` (as BARRIER reads
        them) and a ``report`` with one row per section: ``x``, ``km`` (to the mouth), ``bed``, ``bed_offset`` (m
        from the axis), ``width`` and ``area`` (at ``check_level``), ``area_jump`` (change of that area from the
        previous section, |Δ ln A|), ``open_end`` (at ``top_level``) and ``crosses_next`` (at ``check_level``).
    """
    if sections is None:
        if axis is None or dx is None:
            raise ValueError("Give 'axis' and 'dx', or 'sections'")
        sections = section_lines(axis, dx, axis_smooth)
    sections = sections.reset_index(drop=True)
    if half_width is None or top_level is None:
        raise ValueError("Give 'half_width' and 'top_level'")
    s, prof = sample_profiles(dem, sections, half_width, profile_step)
    bed_hw = np.broadcast_to(np.asarray(bed_half_width if bed_half_width is not None else np.inf, dtype=float),
                             (len(sections),))
    tabs = [section_table(s, prof[i], x, level_step, top_level, None if np.isinf(bed_hw[i]) else bed_hw[i])
            for i, x in enumerate(sections["x"])]
    raw = pd.concat(tabs, ignore_index=True)
    open_end = raw.groupby("x", sort=True)["open_end"].last()
    bed_offset = pd.Series([t["xl"].iloc[0] for t in tabs], index=sections["x"])
    cs = copy_downstream(raw[CROSS_SECTION_COLUMNS], downstream_copies)
    if smooth:
        cs = copy_downstream(smooth_sections(cs, smooth, level_step), downstream_copies)
    cs = cs.sort_values(["x", "eta"], kind="mergesort").reset_index(drop=True)

    return _finish(cs, sections, top_level, check_level, nmann, kh, out_dir, verbose, open_end, bed_offset)


def _along_values(x, spec, name):
    """Value of a parameter at each section: a number, or a dict of options of
    :func:`~environmentaltools.models.barrier.set_section_parameter` (``points``, ``steps``, ``coordinate``…)."""
    if isinstance(spec, dict):
        from .inputs import set_section_parameter

        return set_section_parameter(pd.DataFrame({"x": x}), name, **spec)[name].to_numpy(dtype=float)
    return np.full(len(x), float(spec))


def prismatic_geometry(length=None, dx=None, bottom_width=None, side_slope=0.0, bed=None, top_level=None,
                       convergence_length=None, axis=None, sections=None, level_step=0.05, downstream_copies=0,
                       nmann=0.03, kh=None, check_level=None, out_dir=None, verbose=True):
    """Geometry of an idealised channel (rectangular or trapezoidal sections), without a DEM.

    The table of each section is analytic: with water depth ``h``, bottom width ``b`` and side slope ``m``
    (horizontal : vertical; 0 is rectangular), ``A = b h + m h²``, ``B = b + 2 m h``, ``P = b + 2 h √(1 + m²)``,
    ``Rh = A / P``; the banks are symmetric about the axis (``xl = −B/2``, ``xr = B/2``).

    Each of ``bottom_width``, ``side_slope`` and ``bed`` is one number or a dict with the options of
    :func:`~environmentaltools.models.barrier.set_section_parameter`, e.g. a bed that deepens towards the mouth,
    ``{"points": [[0, -8], [60, -3]], "coordinate": "km"}`` (km to the mouth), or a width by stretches
    (``steps``).

    Args:
        length (float | None): Channel length (m), for a straight axis from ``(0, 0)`` eastwards (the mouth at
            ``(length, 0)``); not needed with ``axis`` or ``sections``.
        dx (float): Distance between sections (m).
        bottom_width (float | dict): Bottom width ``b`` (m).
        side_slope (float | dict): Side slope ``m`` (horizontal / vertical); 0 = rectangular.
        bed (float | dict): Bed elevation (m, same datum as the tide).
        top_level (float): Highest level of the tables (m).
        convergence_length (float | None): Exponential convergence of the width, as in an idealised estuary:
            ``b`` is multiplied by ``exp(−d / convergence_length)``, with ``d`` the distance to the mouth (m), so
            ``bottom_width`` is the width at the mouth.
        axis: Channel axis instead of the straight one (see :func:`read_axis`), e.g. to place the channel on a map.
        sections (pd.DataFrame | None): Section lines instead of ``length``/``axis`` and ``dx`` (see
            :func:`build_geometry`).
        level_step (float): Vertical step of the tables (m).
        downstream_copies, nmann, kh, check_level, out_dir: See :func:`build_geometry`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ``cross_sections``, ``along_channel`` and ``report``, as
        :func:`build_geometry`.
    """
    if sections is None:
        if dx is None:
            raise ValueError("Give 'dx' (or 'sections')")
        if axis is None:
            if length is None:
                raise ValueError("Give 'length', 'axis' or 'sections'")
            axis = [[0.0, 0.0], [float(length), 0.0]]
        sections = section_lines(axis, dx)
    sections = sections.reset_index(drop=True)
    if bottom_width is None or bed is None or top_level is None:
        raise ValueError("Give 'bottom_width', 'bed' and 'top_level'")
    x = sections["x"].to_numpy(dtype=float)
    b = _along_values(x, bottom_width, "bottom_width")
    if convergence_length:
        b = b * np.exp(-distance_to_mouth(x) * 1000.0 / convergence_length)
    m = _along_values(x, side_slope, "side_slope")
    z = _along_values(x, bed, "bed")
    if np.any(b < 0) or np.any(m < 0) or np.any((b == 0) & (m == 0)):
        raise ValueError("bottom_width and side_slope must be >= 0 and not both 0")
    if np.any(z >= top_level):
        raise ValueError(f"top_level ({top_level:g}) must be above the bed of every section (max {z.max():g})")
    rows = []
    for xi, bi, mi, zi in zip(x, b, m, z):
        h = np.arange(0.0, top_level - zi + level_step / 2, level_step)
        A, B, P = bi * h + mi * h ** 2, bi + 2 * mi * h, bi + 2 * h * np.sqrt(1 + mi ** 2)
        rows.append(pd.DataFrame(dict(x=xi, z=zi, eta=h, B=B, A=A, P=P,
                                      Rh=np.divide(A, P, out=np.zeros_like(A), where=P > 0), sigma=B, xl=-B / 2,
                                      xr=B / 2, beta=1)))
    cs = copy_downstream(pd.concat(rows, ignore_index=True), downstream_copies)
    cs = cs.sort_values(["x", "eta"], kind="mergesort").reset_index(drop=True)
    return _finish(cs, sections, top_level, check_level, nmann, kh, out_dir, verbose)


def plot_geometry(cs, along, dem=None, examples=4, top_level=None, file_name=None):
    """Plan view of the sections (wet line at the top level, over the DEM), bed and width along the channel and a few
    sections as BARRIER sees them (banks ``xl``, ``xr`` at each level).

    Args:
        cs (pd.DataFrame): Cross-section tables.
        along (pd.DataFrame): Along-channel data (``x``, ``xutm``, ``yutm``, ``AngMd``, ``AngMi``).
        dem (str | Path | None): Raster drawn below the plan view.
        examples (int): Number of example sections (evenly spaced from the upstream boundary to the mouth).
        top_level (float | None): Level of the wet lines and widths; ``None``: the top of the tables.
        file_name (str | Path | None): Save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt

    tables = dict(tuple(cs.groupby("x", sort=True)))
    xs = along["x"].to_numpy()
    top = (cs["z"] + cs["eta"]).max() if top_level is None else top_level
    km = distance_to_mouth(xs)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, examples, height_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    if dem is not None and not _is_xyz(dem):
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(dem) as src:
            f = max(1, int(max(src.width, src.height) / 1500))
            img = src.read(1, out_shape=(src.height // f, src.width // f), resampling=Resampling.average,
                           masked=True)
            b = src.bounds
        im = ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top), cmap="terrain", origin="upper")
        fig.colorbar(im, ax=ax, label="DEM (m)", shrink=0.8)
    for _, sec in along.iterrows():
        t = tables[sec["x"]]
        xl, xr = _value_at(t, top, "xl"), _value_at(t, top, "xr")
        px, py = section_points(sec.to_frame().T, [xl, 0.0, xr])
        ax.plot(px[0], py[0], color="#2a78d6", lw=0.6)
    ax.plot(along["xutm"], along["yutm"], color="#1a1a1a", lw=1.0)
    ax.annotate("upstream", (along["xutm"].iloc[0], along["yutm"].iloc[0]), fontsize=8)
    ax.annotate("mouth", (along["xutm"].iloc[-1], along["yutm"].iloc[-1]), fontsize=8)
    pad = 0.05 * max(np.ptp(along["xutm"]), np.ptp(along["yutm"]), 1.0)
    w = max(abs(cs["xl"]).max(), abs(cs["xr"]).max())
    ax.set_xlim(along["xutm"].min() - w - pad, along["xutm"].max() + w + pad)
    ax.set_ylim(along["yutm"].min() - w - pad, along["yutm"].max() + w + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Sections: wet line at {top:g} m", loc="left")

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(km, [tables[x]["z"].iloc[0] for x in xs], color="#1a1a1a", lw=1.2, label="bed")
    ax1.set_ylabel("bed (m)")
    ax2 = ax1.twinx()
    ax2.plot(km, [_value_at(tables[x], top, "B") for x in xs], color="#2a78d6", lw=1.0)
    ax2.set_ylabel(f"width at {top:g} m (m)", color="#2a78d6")
    ax1.set_xlabel("km to the mouth")
    ax1.invert_xaxis()

    for k, i in enumerate(np.linspace(0, len(xs) - 1, examples).round().astype(int)):
        axk = fig.add_subplot(gs[2, k])
        t = tables[xs[i]].sort_values("eta")
        lev = (t["z"] + t["eta"]).to_numpy()
        axk.plot(np.r_[t["xl"].to_numpy()[::-1], t["xr"].to_numpy()], np.r_[lev[::-1], lev], color="#1a1a1a", lw=1.2)
        axk.set_title(f"km {km[i]:.2f}", loc="left", fontsize=9)
        axk.set_xlabel("m from the axis (left < 0)", fontsize=8)
    fig.tight_layout()
    if file_name is not None:
        fig.savefig(file_name, dpi=150)
    return fig


#: Function of each ``type`` of the ``geometry`` section of a project file.
GEOMETRY_TYPES = {"dem": "build_geometry", "prismatic": "prismatic_geometry"}
#: Keys of the ``geometry`` section that are files (relative to the project file).
GEOMETRY_FILES = ("dem", "axis")


def geometry_settings(project):
    """Function and arguments of the ``geometry`` section of a project file, with its files resolved.

    Returns:
        tuple: ``(type, function, kwargs)``.

    Raises:
        ValueError: If ``type`` is missing or unknown, or a key is not an argument of the function.
    """
    import inspect

    cfg = dict(project["geometry"] or {})
    kind = cfg.pop("type", None)
    if kind not in GEOMETRY_TYPES:
        raise ValueError(f"geometry: 'type' must be one of {sorted(GEOMETRY_TYPES)} (got {kind!r})")
    func = globals()[GEOMETRY_TYPES[kind]]
    allowed = set(inspect.signature(func).parameters) - {"sections", "out_dir", "verbose"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"geometry ({kind}): unknown keys {sorted(unknown)}; use {sorted(allowed)}")
    for k in GEOMETRY_FILES:
        if isinstance(cfg.get(k), str):
            cfg[k] = project.path(cfg[k])
    return kind, func, cfg


def _geometry_files(base):
    """Names of the geometry files of the base case: those of its ``config.yaml``, else the usual ones."""
    names = {"cross_sections_file": "cross-sections.csv", "along_channel_file": "along_channel_data.csv"}
    if (Path(base) / "config.yaml").exists():
        from .config import read_config

        names.update({k: v for k, v in (read_config(base).get("geometry") or {}).items() if k in names and v})
    return names


def _sha1(path):
    import hashlib

    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _canon(d):
    import json

    return json.dumps(d, sort_keys=True, default=str)


def _stamp(files):
    """Size and date of input files (a DEM can be large: its content is not read)."""
    return {k: [str(f), f.stat().st_size, f.stat().st_mtime_ns] for k, f in files.items()}


def pending(signature, settings, inputs, base, outputs, report_dir=None):
    """Why the files that a section of the project file writes in the base case must be written again.

    Args:
        signature (Path): JSON written by :func:`save_signature` the last time.
        settings (dict): The section.
        inputs (dict): ``{key: Path}`` of its input files (size and date are compared).
        base (Path): Base case folder.
        outputs (list[str]): Files of the base case written by the section.
        report_dir (Path | None): A file edited by hand is copied to ``report_dir/edited`` before it is replaced.

    Returns:
        str | None: The reason, or None if everything is up to date.
    """
    import json
    import shutil

    if not signature.exists():
        return "not written yet"
    saved = json.loads(signature.read_text())
    if _canon(saved.get("settings")) != _canon(settings):
        return "the section changed"
    stamps = _stamp(inputs)
    for k in sorted(set(stamps) | set(saved.get("inputs") or {})):
        if _canon(stamps.get(k)) != _canon((saved.get("inputs") or {}).get(k)):
            return f"the file of '{k}' changed"
    for name in outputs:
        f = Path(base) / name
        if not f.exists():
            return f"{name} is missing in the base case"
        if _sha1(f) != (saved.get("outputs") or {}).get(name):
            if report_dir is not None:
                (Path(report_dir) / "edited").mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, Path(report_dir) / "edited" / name)
                return f"{name} of the base case was edited by hand (copy in {Path(report_dir) / 'edited'})"
            return f"{name} of the base case was edited by hand"
    return None


def save_signature(signature, settings, inputs, base, outputs):
    """Fingerprint of a section of the project file, its input files and the files it wrote (see :func:`pending`)."""
    import json

    Path(signature).parent.mkdir(parents=True, exist_ok=True)
    Path(signature).write_text(json.dumps({"settings": json.loads(_canon(settings)), "inputs": _stamp(inputs),
                                           "outputs": {n: _sha1(Path(base) / n) for n in outputs}}, indent=1))


def prepare_geometry(project, force=False, verbose=True):
    """Build the geometry of the base case from the ``geometry`` section of the project file, if it is pending.

    The section holds ``type`` (``dem``: :func:`build_geometry`; ``prismatic``: :func:`prismatic_geometry`) and the
    arguments of that function, with the files (``dem``, ``axis``) relative to the project file::

        geometry:
          type: dem
          dem: inputs/dem.tif
          axis: inputs/axis.shp
          dx: 200
          half_width: 2000
          top_level: 5
          smooth: 500

    ``cross-sections.csv`` and ``along_channel_data.csv`` are written in the base case (with the names of its
    ``config.yaml``, if it has one), so :func:`~environmentaltools.models.barrier.prepare` then creates the cases
    again. ``base_report_dir`` gets ``geometry.png`` (:func:`plot_geometry`), ``geometry_report.xlsx`` (the report)
    and ``geometry_signature.json``. The geometry is built again only when the section, the DEM or the axis (size
    and date) changed, or when the files of the base case are missing or were edited by hand (see :func:`pending`).

    Args:
        project (BarrierProject): Project configuration.
        force (bool): Build it again anyway.

    Returns:
        pd.DataFrame | None: The report, or None if there was nothing to do.
    """
    if not project["geometry"]:
        return None
    kind, func, kwargs = geometry_settings(project)
    base, report_dir = project.base_case, project.path(project["base_report_dir"])
    names = _geometry_files(base)
    inputs = {k: kwargs[k] for k in GEOMETRY_FILES if isinstance(kwargs.get(k), Path)}
    signature = report_dir / "geometry_signature.json"
    why = "forced" if force else pending(signature, project["geometry"], inputs, base, list(names.values()),
                                         report_dir)
    if why is None:
        if verbose:
            print(f"Geometry: up to date ({base})")
        return None
    if verbose:
        print(f"Geometry ({kind}): {why} → building it")

    cs, along, report = func(**kwargs, verbose=verbose)
    base.mkdir(parents=True, exist_ok=True)
    write_csv(cs, base / names["cross_sections_file"])
    write_csv(along, base / names["along_channel_file"])
    report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    from ._xlsx import write_tables

    plot_geometry(cs, along, dem=kwargs.get("dem") if kind == "dem" else None, top_level=kwargs.get("check_level"),
                  file_name=report_dir / "geometry.png")
    plt.close("all")
    write_tables(report_dir / "geometry_report.xlsx", {"sections": report}, legend={
        "x": "Section: m from the upstream boundary (as in the BARRIER files)",
        "km": "Distance to the mouth (km)",
        "bed": "Bed elevation (m)",
        "bed_offset": "Position of the bed point (m from the axis, left < 0)",
        "width, area": "Width B (m) and area A (m²) at check_level (top_level if not given)",
        "area_jump": "Change of the area from the previous section, |Δ ln A| (large jumps: use smooth)",
        "open_end": "The water reaches the end of the profile or a gap of the DEM at top_level",
        "crosses_next": "The wet line at check_level crosses the next section",
        "files": f"{names['cross_sections_file']} and {names['along_channel_file']} in {base}",
    })
    save_signature(signature, project["geometry"], inputs, base, list(names.values()))
    return report


def prepare_base(project, force=False, verbose=True):
    """Step ``base`` of the workflow: the base case from the sections ``geometry`` and ``simulation`` of the
    project file, each one only if it is in the project and something changed (:func:`prepare_geometry`,
    :func:`~environmentaltools.models.barrier.prepare_simulation`). Without either section the base case is used
    as it is.

    Args:
        project (BarrierProject): Project configuration.
        force (bool): Write both again anyway.

    Returns:
        bool: True if a file of the base case was written.
    """
    from .base_case import prepare_simulation

    if not project["geometry"] and not project["simulation"]:
        if verbose:
            print("No 'geometry' nor 'simulation' section in the project file: the base case is used as it is")
        return False
    built = prepare_geometry(project, force, verbose) is not None
    return prepare_simulation(project, force, verbose) or built
