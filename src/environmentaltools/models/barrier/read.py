"""Readers of BARRIER results: NetCDF output, series at points, tidal mean and comparison of runs."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: Default variables of :func:`compare`: ``{variable: tolerance}`` (largest difference taken as "equal").
COMPARE_TOLERANCES = {"eta": 0.005, "Q": 1.0, "S": 0.05, "A": 1.0}


def latest_output(folder):
    """Newest NetCDF output in ``folder``.

    Raises:
        FileNotFoundError: If there is none.
    """
    ncs = sorted(Path(folder).glob("*.nc"), key=os.path.getmtime)
    if not ncs:
        raise FileNotFoundError(f"No BARRIER output (*.nc) in {folder}")
    return ncs[-1]


def distance_to_mouth(x):
    """Distance (km) from each section to the mouth (downstream boundary, the section with the largest ``x``)."""
    x = np.asarray(x, dtype=float)
    return (x.max() - x) / 1000.0


def read_output(nc):
    """Open a BARRIER NetCDF output.

    Adds the coordinates ``date`` (on ``time``, from the global attribute
    ``time_offset``) and ``km`` (on ``x``, distance to the mouth in km).

    Args:
        nc (str | Path): NetCDF file, or a folder (its newest output is read).

    Returns:
        xr.Dataset: Variables with dimensions ``(time, x)`` (``eta``, ``Q``, ``S``, ``A``…).
    """
    nc = Path(nc)
    if nc.is_dir():
        nc = latest_output(nc)
    ds = xr.open_dataset(nc)
    start = pd.Timestamp(ds.attrs["time_offset"])
    ds = ds.assign_coords(date=("time", start + pd.to_timedelta(ds["time"].values, unit="s")),
                          km=("x", distance_to_mouth(ds["x"].values)))
    ds.attrs["source_file"] = str(nc)
    return ds


def time_step_hours(ds):
    """Output time step of ``ds`` in hours."""
    return float(np.median(np.diff(ds["time"].values))) / 3600.0


def tidal_mean(values, window_hours=25, step_hours=1.0):
    """Centred moving mean over ``window_hours`` along the first axis (removes the tidal oscillation).

    The 25-h window removes the semidiurnal and diurnal tides and leaves the subtidal
    signal. Values are NaN at the ends, where the window does not fit.

    Args:
        values (array_like | pd.DataFrame): Series with time along the first axis.
        window_hours (float): Window length in hours.
        step_hours (float): Time step of ``values`` in hours.

    Returns:
        np.ndarray: Filtered values, same shape as ``values``.
    """
    window = max(1, int(round(window_hours / step_hours)))
    arr = np.asarray(values, dtype=float)
    out = pd.DataFrame(arr.reshape(len(arr), -1)).rolling(window, center=True).mean().to_numpy()
    return out.reshape(arr.shape)


def nearest_sections(ds, km):
    """Index of the section closest to each distance to the mouth ``km``."""
    km_model = ds["km"].values if "km" in ds.coords else distance_to_mouth(ds["x"].values)
    return [int(np.argmin(np.abs(km_model - k))) for k in np.atleast_1d(km)]


def extract_points(ds, points, variables=("eta", "S")):
    """Time series of BARRIER variables at points along the channel.

    Each point takes the section closest to its distance to the mouth.

    Args:
        ds (xr.Dataset): Output read with :func:`read_output`.
        points (pd.DataFrame | dict): Columns ``name`` and ``km`` (distance to the mouth),
            or ``{name: km}``.
        variables (list[str]): Variables with dimensions ``(time, x)``.

    Returns:
        tuple[pd.DataFrame, dict]: Points (``name``, ``km``, ``section``, ``km_section`` and,
        if the output has the bank coordinates, ``utm_x``/``utm_y`` of the section centre
        averaged in time) sorted from the mouth upstream, and ``{variable: DataFrame}``
        indexed by ``date`` with one column ``"<name> (km <km_section>)"`` per point.
    """
    if isinstance(points, dict):
        points = pd.DataFrame({"name": list(points), "km": list(points.values())})
    points = points.dropna(subset=["name", "km"]).sort_values("km").reset_index(drop=True)
    idx = nearest_sections(ds, points["km"].to_numpy(dtype=float))
    points["section"] = idx
    points["km_section"] = ds["km"].values[idx].round(2)
    for axis in ("x", "y"):
        if f"xl_utm_{axis}" in ds and f"xr_utm_{axis}" in ds:
            centre = (ds[f"xl_utm_{axis}"] + ds[f"xr_utm_{axis}"]) / 2   # banks move with the level
            points[f"utm_{axis}"] = centre.isel(x=idx).mean("time").values.round(1)
    columns = [f"{n} (km {k:g})" for n, k in zip(points["name"], points["km_section"])]
    dates = pd.DatetimeIndex(ds["date"].values, name="date")
    series = {var: pd.DataFrame(ds[var].isel(x=idx).values, index=dates, columns=columns) for var in variables}
    return points, series


def compare(ref, new, tolerances=None):
    """Compare two BARRIER outputs over their common time steps (e.g. a new run against a reference).

    Args:
        ref (xr.Dataset): Reference output.
        new (xr.Dataset): Output compared with it (same sections).
        tolerances (dict | None): ``{variable: tolerance}`` (default :data:`COMPARE_TOLERANCES`).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Summary per variable (``rmse``, ``max_abs_diff``,
        ``tolerance``, ``within_tolerance``, ``nan_steps_new``, ``steps``) and RMSE per
        section (``section``, ``km``, ``rmse_<variable>``).
    """
    tolerances = tolerances or COMPARE_TOLERANCES
    if ref.sizes["x"] != new.sizes["x"]:
        raise ValueError(f"Different number of sections: {ref.sizes['x']} and {new.sizes['x']}")
    n = min(ref.sizes["time"], new.sizes["time"])
    rows, per_section = [], {}
    for var, tol in tolerances.items():
        a = ref[var].isel(time=slice(0, n)).values
        b = new[var].isel(time=slice(0, n)).values
        diff = b - a
        max_abs = float(np.nanmax(np.abs(diff)))
        rows.append({"variable": var, "units": ref[var].attrs.get("units", ""),
                     "rmse": float(np.sqrt(np.nanmean(diff ** 2))), "max_abs_diff": max_abs,
                     "tolerance": tol, "within_tolerance": max_abs <= tol,
                     "nan_steps_new": int(np.isnan(b).any(axis=1).sum()), "steps": n})
        per_section[f"rmse_{var}"] = np.sqrt(np.nanmean(diff ** 2, axis=0))
    km = ref["km"].values if "km" in ref.coords else distance_to_mouth(ref["x"].values)
    sections = pd.DataFrame({"section": np.arange(ref.sizes["x"]), "km": km.round(2), **per_section})
    return pd.DataFrame(rows), sections


def read_project(project, variables=None):
    """Outputs of every finished case of a project.

    Args:
        project (BarrierProject): Project configuration.
        variables (list[str] | None): Keep only these variables.

    Returns:
        dict: ``{case: xr.Dataset}``.
    """
    out = {}
    for case in project.case_names:
        res = project.results_dir / case
        if (res / "done.json").exists():
            ds = read_output(res)
            out[case] = ds[variables] if variables else ds
    return out
