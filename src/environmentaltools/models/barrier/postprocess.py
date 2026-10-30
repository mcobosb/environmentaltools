"""Automatic postprocessing of the cases of a BARRIER project: series at points, figures and comparison.

Everything is set in the ``postprocess`` section of the project file; see
``barrier_template.yaml``. For each finished case (and for an external reference
output, if there is one) the folder ``postprocess_dir/<case>/`` gets:

- ``timeseries.xlsx``: one sheet per variable (one column per point) and the points used;
- ``timeseries_<variable>.png``: one panel per point, raw series and tidal mean;
- ``along_channel_<variable>.png``: the variable in every section against time;
- with a ``reference``: ``vs_<reference>.xlsx`` (differences per variable and per
  section) and ``vs_<reference>.png`` (level and salinity at a few stations).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .read import COMPARE_TOLERANCES, compare, extract_points, read_output

TIMESERIES = "timeseries.xlsx"   # written last: marks a case as postprocessed


def load_points(project):
    """Points of the series (``postprocess: points``) as a DataFrame with columns ``name`` and ``km``.

    ``points`` can be a file (xlsx or csv), a list of ``{name, km}`` or ``{name: km}``. The
    columns of a file are mapped with ``points_columns`` (e.g. ``{name: nombre, km: km_desde_desembocadura}``);
    other columns are kept.

    Returns:
        pd.DataFrame | None: The points, or ``None`` if there are none.
    """
    pp = project["postprocess"]
    pts = pp.get("points")
    if not pts:
        return None
    if isinstance(pts, str):
        f = project.path(pts)
        df = pd.read_excel(f) if f.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(f)
        cols = pp.get("points_columns") or {}
        df = df.rename(columns={cols.get("name", "name"): "name", cols.get("km", "km"): "km"})
    elif isinstance(pts, dict):
        df = pd.DataFrame({"name": list(pts), "km": list(pts.values())})
    else:
        df = pd.DataFrame(pts)
    if not {"name", "km"} <= set(df.columns):
        raise ValueError("postprocess: points need the columns 'name' and 'km' (see points_columns)")
    return df.dropna(subset=["name", "km"]).reset_index(drop=True)


def _label(ds, var):
    a = ds[var].attrs
    name = a.get("long_name", var)
    return f"{name} ({a['units']})" if a.get("units") else name


def _stations(ds, stations_km):
    """Default stations of the comparison figure: four sections from the mouth to 80 % of the channel."""
    if stations_km:
        return list(stations_km)
    return list(np.round(np.linspace(0, 0.8 * float(ds["km"].max()), 4)))


def postprocess_output(ds, label, out_dir, points=None, settings=None, reference=None):
    """Series at points, figures and (optionally) comparison of one BARRIER output.

    Args:
        ds (xr.Dataset): Output read with :func:`~environmentaltools.models.barrier.read_output`.
        label (str): Name of the case (titles).
        out_dir (str | Path): Output folder (created if needed).
        points (pd.DataFrame | None): Columns ``name`` and ``km`` (see :func:`load_points`).
        settings (dict | None): ``postprocess`` section of the project (defaults of
            :data:`~environmentaltools.models.barrier.config.DEFAULTS`).
        reference (tuple[str, xr.Dataset] | None): ``(name, output)`` to compare with.

    Returns:
        pd.DataFrame | None: Comparison summary (with a reference), else ``None``.
    """
    import matplotlib.pyplot as plt

    from ._xlsx import write_tables
    from .config import DEFAULTS
    from .plot import plot_along_channel, plot_comparison, plot_point_series

    s = {**DEFAULTS["postprocess"], **(settings or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(ds.attrs.get("source_file", "")).name
    notes = s.get("legend_notes") or {}
    summary = None

    if reference is not None:
        ref_name, ref = reference
        summary, sections = compare(ref, ds, s.get("tolerances") or COMPARE_TOLERANCES)
        runs = pd.DataFrame({"item": ["case", "output", "reference", "reference output", "BARRIER version (case)",
                                      "BARRIER version (reference)", "time steps compared"],
                             "value": [label, source, ref_name, Path(ref.attrs.get("source_file", "")).name,
                                       ds.attrs.get("Version", ""), ref.attrs.get("Version", ""),
                                       int(summary["steps"].iloc[0])]})
        write_tables(out_dir / f"vs_{ref_name}.xlsx", {"summary": summary, "per_section": sections, "runs": runs},
                     legend={
                         "rmse": f"Root mean square difference ({label} − {ref_name}) over every section and time step",
                         "max_abs_diff": "Largest absolute difference in any section and time step",
                         "tolerance": "Largest difference taken as 'equal'",
                         "within_tolerance": "True if max_abs_diff <= tolerance",
                         "nan_steps_new": "Time steps with some NaN in the case (should be 0)",
                         "sheet per_section": "RMSE of each variable in each section (km to the mouth)",
                         "sheet runs": "Outputs and BARRIER versions compared", **notes})
        plot_comparison(ref, ds, _stations(ds, s.get("stations_km")), s.get("zoom_days"),
                        labels=(ref_name, label), window_hours=s.get("window_hours"),
                        title=f"{label} vs {ref_name}", file_name=out_dir / f"vs_{ref_name}.png")

    variables = [v for v in s["variables"] if v in ds]
    along = s.get("along_channel") or {}
    if along.get("variable") in ds:
        var = along["variable"]
        wh = s.get("window_hours")
        plot_along_channel(ds, var, None if points is None else extract_points(ds, points, [])[0],
                           along.get("levels"), wh,
                           label=f"{'tidal-mean ' if wh else ''}{_label(ds, var)}",
                           title=f"{_label(ds, var).capitalize()} along the channel · {label}",
                           file_name=out_dir / f"along_channel_{var}.png")
    if points is not None and variables:
        pts, series = extract_points(ds, points, variables)
        for var in variables:
            plot_point_series(series[var], _label(ds, var), f"{_label(ds, var).capitalize()} · {label}",
                              s.get("window_hours"), file_name=out_dir / f"timeseries_{var}.png")
        sheets = {var: series[var].round(4).reset_index() for var in variables}
        write_tables(out_dir / TIMESERIES, {**sheets, "points": pts}, legend={
            **{f"sheet {v}": f"{_label(ds, v)} at each point and output time" for v in variables},
            "date": f"Date and time (start of the simulation: {ds.attrs.get('time_offset', '')})",
            "columns": "One per point: 'name (km to the mouth of the section used)'",
            "sheet points": "Points asked for and section of the model used (the closest one)",
            "section": "Index of the section (0 = upstream boundary, last = mouth)",
            "utm_x, utm_y": "Centre of the section (mean of both banks and of time), coordinates of the model",
            "output": f"{source} (BARRIER {ds.attrs.get('Version', '')})", **notes})
    else:   # no points: leave the marker anyway so the case counts as postprocessed
        write_tables(out_dir / TIMESERIES, {"info": pd.DataFrame({"item": ["output"], "value": [source]})})
    plt.close("all")
    return summary


def _is_up_to_date(out_dir, sources):
    marker = Path(out_dir) / TIMESERIES
    if not marker.exists():
        return False
    t = marker.stat().st_mtime
    return all(t > Path(f).stat().st_mtime for f in sources if f and Path(f).exists())


def postprocess(project, cases=None, force=False, verbose=True):
    """Postprocess the finished cases of a project that are new or were run again.

    A case is processed when its folder ``postprocess_dir/<case>/`` is older than its
    result, the points file or the project file (so changing the points or the
    settings processes everything again). An external reference output
    (``reference: {file: …}``) is processed too.

    Args:
        project (BarrierProject): Project configuration.
        cases (list[str] | None): Only these cases.
        force (bool): Process again every finished case.

    Returns:
        pd.DataFrame: One row per case: ``case``, ``status`` (done / up to date / no result) and,
        with a reference, the RMSE of each variable.
    """
    s = project["postprocess"]
    points = load_points(project)
    watched = [project.file, project.path(s["points"]) if isinstance(s.get("points"), str) else None]
    ref_cfg = s.get("reference") or {}
    ref = None
    rows = []
    if ref_cfg.get("file"):
        name = ref_cfg.get("name") or Path(ref_cfg["file"]).stem
        f = project.path(ref_cfg["file"])
        ref = (name, read_output(f))
        out = project.postprocess_dir / name
        if force or not _is_up_to_date(out, watched + [f]):
            postprocess_output(ref[1], name, out, points, s)
            rows.append({"case": name, "status": "done (reference)"})
        else:
            rows.append({"case": name, "status": "up to date (reference)"})
    elif ref_cfg.get("case"):
        done = project.results_dir / ref_cfg["case"] / "done.json"
        if done.exists():
            ref = (ref_cfg.get("name") or ref_cfg["case"], read_output(project.results_dir / ref_cfg["case"]))
            watched.append(done)          # a new reference result makes every comparison outdated

    for case in cases or project.case_names:
        done = project.results_dir / case / "done.json"
        out = project.postprocess_dir / case
        if not done.exists():
            rows.append({"case": case, "status": "no result"})
            continue
        if not force and _is_up_to_date(out, watched + [done]):
            rows.append({"case": case, "status": "up to date"})
            continue
        ds = read_output(project.results_dir / case)
        use_ref = ref if ref is not None and ref_cfg.get("case") != case else None
        summary = postprocess_output(ds, case, out, points, s, use_ref)
        row = {"case": case, "status": "done"}
        if summary is not None:
            row.update({f"rmse_{v}": r for v, r in zip(summary["variable"], summary["rmse"])})
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print(df.to_string(index=False))
        print(f"Results in {project.postprocess_dir}/<case>/")
    return df
