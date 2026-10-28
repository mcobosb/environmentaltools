"""BARRIER input files: CSV tables, boundary series, tributaries and along-channel parameters.

BARRIER reads comma-separated files whose header line starts with ``;`` (lines
starting with ``;`` or ``#`` are comments). Section ``x`` grows from the
upstream boundary (``x = 0``) to the downstream boundary (the mouth, last section).
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: Columns of ``cross-sections.csv``: one row per section ``x`` and water depth ``eta``.
CROSS_SECTION_COLUMNS = ["x", "z", "eta", "B", "A", "P", "Rh", "sigma", "xl", "xr", "beta"]
#: Direction expected for each variable when the depth grows: +1 increasing, -1 decreasing.
MONOTONIC = {"B": 1, "A": 1, "P": 1, "xl": -1, "xr": 1}


def read_csv(file_name):
    """Read a BARRIER CSV table (header line starting with ``;``).

    Args:
        file_name (str | Path): CSV file.

    Returns:
        pd.DataFrame: Table with clean column names.
    """
    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip(" ;#") for c in df.columns]
    return df


def write_csv(df, file_name, float_format=None):
    """Write a table as BARRIER expects it: comma-separated, header line starting with ``;``.

    Args:
        df (pd.DataFrame): Table (the index is not written).
        file_name (str | Path): Output file.
        float_format (str | None): Number format; ``None`` writes every digit (shortest
            round-trip representation). Keep it: rounding the geometry or the along-channel
            data (e.g. ``z = -4.2700000000000005`` → ``-4.27``) changes BARRIER results by decimetres.

    Returns:
        Path: Written file.
    """
    file_name = Path(file_name)
    with open(file_name, "w", encoding="utf-8", newline="\n") as f:
        f.write(";" + ",".join(map(str, df.columns)) + "\n")
        df.to_csv(f, header=False, index=False, float_format=float_format, lineterminator="\n")
    return file_name


def seconds_since(index, start):
    """Seconds since ``start`` of a datetime index (numeric indexes are taken as seconds already)."""
    if isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index):
        return (pd.DatetimeIndex(index) - pd.Timestamp(start)).total_seconds().to_numpy()
    return np.asarray(index, dtype=float)


def write_series(series, file_name, start=None, names=("time_seconds", "value"), float_format=None):
    """Write a boundary time series (tide, discharge, temperature…) as ``seconds, value``.

    Args:
        series (pd.Series): Values indexed by date (needs ``start``) or by seconds since the start.
        file_name (str | Path): Output file.
        start (str | datetime | None): Start of the simulation (``run: start_date`` of the config).
        names (tuple[str, str]): Names of the header line.
        float_format (str | None): Number format (``None``: every digit).

    Returns:
        Path: Written file.
    """
    series = series.dropna()
    if start is None and pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("Give 'start' (simulation start date) for a series indexed by date")
    t = seconds_since(series.index, start)
    if np.any(np.diff(t) <= 0):
        raise ValueError("The series must be sorted in time without repeated times")
    if t[0] > 0:
        print(f"Warning: {Path(file_name).name} starts {t[0]:.0f} s after the start of the simulation")
    return write_csv(pd.DataFrame({names[0]: t, names[1]: series.to_numpy(dtype=float)}), file_name,
                     float_format)


def write_tributaries(tributaries, file_name, start=None, float_format=None):
    """Write the lateral inflows (tributaries) file of BARRIER.

    Each tributary enters at the section closest to its UTM position.

    Args:
        tributaries (list[dict]): One dict per tributary with ``x_utm``, ``y_utm``,
            ``discharge`` (pd.Series in m³/s, indexed by date or by seconds) and,
            optionally, ``name``.
        file_name (str | Path): Output file (``hydrodynamics: forcing: tributaries_file``).
        start (str | datetime | None): Start of the simulation, for series indexed by date.
        float_format (str | None): Number format (``None``: every digit).

    Returns:
        Path: Written file.
    """
    file_name = Path(file_name)
    with open(file_name, "w", encoding="utf-8", newline="\n") as f:
        f.write("; Number of hydrographs\n")
        f.write(f"{len(tributaries)}\n")
        for k, trib in enumerate(tributaries, 1):
            q = trib["discharge"].dropna()
            f.write(f"; Hydrograph {k}: {trib.get('name', '')}\n")
            f.write(f"{trib['x_utm']:.3f},{trib['y_utm']:.3f}\n")
            pd.DataFrame({"t": seconds_since(q.index, start), "q": q.to_numpy(dtype=float)}).to_csv(
                f, header=False, index=False, float_format=float_format, lineterminator="\n")
    return file_name


def section_profile(x, value=None, points=None, steps=None, sigma=None, mode="reflect"):
    """Along-channel profile of a section parameter (Manning n, Kh, storage factor Sf, β…).

    Exactly one of ``value``, ``points`` or ``steps`` gives the raw profile, which
    can then be smoothed with a Gaussian filter.

    Args:
        x (array_like): Section positions (m, from the upstream boundary).
        value (float | None): Constant value.
        points (list[tuple[float, float]] | None): ``(x, value)`` pairs, linearly interpolated
            (constant beyond the ends).
        steps (list[tuple[float, float]] | None): ``(x_from, value)`` pairs, piecewise constant:
            each value applies from its ``x_from`` to the next one (sorted by ``x_from``).
        sigma (float | None): Standard deviation of the Gaussian smoothing, in number of
            sections (``scipy.ndimage.gaussian_filter1d``).
        mode (str): Edge handling of the smoothing (``reflect``, ``nearest``…).

    Returns:
        np.ndarray: Value at each section.
    """
    x = np.asarray(x, dtype=float)
    if sum(v is not None for v in (value, points, steps)) != 1:
        raise ValueError("Give exactly one of 'value', 'points' or 'steps'")
    if value is not None:
        out = np.full(x.shape, float(value))
    elif points is not None:
        xp, fp = np.asarray(points, dtype=float).T
        out = np.interp(x, xp, fp)
    else:
        xs, vs = np.asarray(sorted(steps), dtype=float).T
        out = vs[np.clip(np.searchsorted(xs, x, side="right") - 1, 0, None)]
    if sigma:
        from scipy.ndimage import gaussian_filter1d

        out = gaussian_filter1d(out, sigma=sigma, mode=mode)
    return out


def set_section_parameter(along, column, value=None, points=None, steps=None, sigma=None, mode="reflect",
                          scale=None, add=None):
    """Change a parameter of ``along_channel_data`` (``nmann``, ``kh``, ``Sf``, ``beta``…).

    The new profile comes from :func:`section_profile` (``value``/``points``/``steps``,
    optionally smoothed with ``sigma``) or from the current one, multiplied by
    ``scale`` and/or shifted by ``add``.

    Args:
        along (pd.DataFrame): Along-channel table (column ``x`` in m).
        column (str): Column to change (created if missing).
        value, points, steps, sigma, mode: See :func:`section_profile`.
        scale (float | None): Factor applied to the profile.
        add (float | None): Amount added to the profile.

    Returns:
        pd.DataFrame: Copy of ``along`` with the new column.
    """
    out = along.copy()
    if any(v is not None for v in (value, points, steps)):
        prof = section_profile(out["x"], value, points, steps, sigma, mode)
    else:
        if column not in out:
            raise KeyError(f"Column '{column}' not in the along-channel table; give 'value', 'points' or 'steps'")
        prof = out[column].to_numpy(dtype=float)
        if sigma:
            from scipy.ndimage import gaussian_filter1d

            prof = gaussian_filter1d(prof, sigma=sigma, mode=mode)
    if scale is not None:
        prof = prof * scale
    if add is not None:
        prof = prof + add
    out[column] = prof
    return out


def _monotonic_violations(eta, values, direction, tol):
    """Indexes ``k`` where, with the depth growing from ``k`` to ``k + 1``, ``values`` goes the wrong way."""
    return np.where((np.diff(eta) > tol) & (direction * np.diff(values) < -tol))[0]


def check_cross_sections(cs, variables=None, tol=1e-9):
    """Check that the cross-section tables are monotonic with the water depth.

    When the depth ``eta`` grows, ``B``, ``A``, ``P`` and ``xr`` must not decrease
    and ``xl`` must not increase; non-monotonic tables make the interpolation of
    BARRIER jump between depths.

    Args:
        cs (pd.DataFrame): ``cross-sections`` table (:data:`CROSS_SECTION_COLUMNS`).
        variables (dict | None): ``{variable: +1 | -1}`` (default :data:`MONOTONIC`).
        tol (float): Tolerance.

    Returns:
        pd.DataFrame: One row per violation (``x``, ``variable``, depths and values before
        and after); empty if the tables are monotonic.
    """
    variables = variables or MONOTONIC
    rows = []
    for x, sec in cs.groupby("x", sort=False):
        sec = sec.dropna(subset=["eta", *variables]).sort_values("eta", kind="mergesort")
        eta = sec["eta"].to_numpy(dtype=float)
        for k in np.where(np.diff(eta) < -tol)[0]:
            rows.append({"x": x, "variable": "eta", "eta_prev": eta[k], "eta_next": eta[k + 1],
                         "value_prev": eta[k], "value_next": eta[k + 1]})
        for var, direction in variables.items():
            v = sec[var].to_numpy(dtype=float)
            for k in _monotonic_violations(eta, v, direction, tol):
                rows.append({"x": x, "variable": var, "eta_prev": eta[k], "eta_next": eta[k + 1],
                             "value_prev": v[k], "value_next": v[k + 1]})
    return pd.DataFrame(rows, columns=["x", "variable", "eta_prev", "eta_next", "value_prev", "value_next"])


def fix_cross_sections(cs, variables=None, step=1e-3, tol=1e-9, max_iter=50):
    """Make the cross-section tables monotonic with the depth.

    Each value that goes the wrong way is replaced by the previous one ± ``step``
    (the correction of the BARRIER calibration of M. Cobos, ``02i_check_cross-sections.py``),
    repeating until no violations remain.

    Args:
        cs (pd.DataFrame): ``cross-sections`` table.
        variables (dict | None): ``{variable: +1 | -1}`` (default :data:`MONOTONIC`).
        step (float): Amount added (or subtracted) to the previous value.
        tol (float): Tolerance.
        max_iter (int): Maximum number of passes.

    Returns:
        tuple[pd.DataFrame, int]: Corrected copy of ``cs`` and number of changed values.
    """
    variables = variables or MONOTONIC
    cs = cs.copy()
    changed = 0
    for _ in range(max_iter):
        changed_pass = 0
        for _, sec in cs.groupby("x", sort=False):
            sec = sec.dropna(subset=["eta", *variables]).sort_values("eta", kind="mergesort")
            if len(sec) <= 1:
                continue
            eta_up = np.diff(sec["eta"].to_numpy(dtype=float)) > tol
            for var, direction in variables.items():
                v = sec[var].to_numpy(dtype=float, copy=True)
                n0 = changed_pass
                for i in np.where(eta_up)[0] + 1:
                    if direction * (v[i] - v[i - 1]) < -tol:
                        v[i] = v[i - 1] + direction * step
                        changed_pass += 1
                if changed_pass > n0:
                    cs.loc[sec.index, var] = v
        changed += changed_pass
        if changed_pass == 0:
            break
    return cs, changed
