"""Figures of BARRIER results: series at points, a variable along the channel in time, comparison of runs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .read import nearest_sections, tidal_mean, time_step_hours

# Sequential blue ramp (light to dark) and neutral ink for text and grid
HOURLY, TIDAL_MEAN = "#86b6ef", "#104281"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REF_COLOR, NEW_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def _save(fig, file_name):
    if file_name:
        fig.savefig(file_name, dpi=130, facecolor=SURFACE)
        plt.close(fig)


def plot_point_series(series, ylabel, title=None, window_hours=25, file_name=None):
    """One panel per point (from the mouth on top to upstream), raw series and tidal mean, same y axis.

    Args:
        series (pd.DataFrame): Series indexed by date, one column per point
            (from :func:`~environmentaltools.models.barrier.extract_points`).
        ylabel (str): Label of the variable (e.g. ``"water level (m)"``).
        title (str | None): Figure title.
        window_hours (float | None): Window of the tidal mean; ``None`` draws only the raw series.
        file_name (str | Path | None): Save the figure here (and close it).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    dates = pd.DatetimeIndex(series.index)
    values = series.to_numpy(dtype=float)
    n = values.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.45 * n + 1.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    mean = None
    if window_hours:
        step = (dates[1] - dates[0]).total_seconds() / 3600.0 if len(dates) > 1 else 1.0
        mean = tidal_mean(values, window_hours, step)
    for k, ax in enumerate(axes):
        ax.plot(dates, values[:, k], color=HOURLY, lw=0.6, label="output")
        if mean is not None:
            ax.plot(dates, mean[:, k], color=TIDAL_MEAN, lw=1.8, label=f"tidal mean ({window_hours:g} h)")
        _style(ax)
        ax.text(0.005, 0.93, series.columns[k], transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    fig.supylabel(ylabel, color=INK, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="upper right", ncol=2)
    if title:
        fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    _save(fig, file_name)
    return fig


def _merged_labels(points, min_km):
    """Tick positions and labels of the points; points closer than ``min_km`` share one label."""
    ticks, labels = [], []
    for name, km in zip(points["name"], points["km_section"]):
        if ticks and km - ticks[-1] < min_km:
            labels[-1] = labels[-1].replace(" · km", f" / {name} · km")
        else:
            ticks.append(km)
            labels.append(f"{name} · km {km:g}")
    return ticks, labels


def plot_along_channel(ds, variable="S", points=None, levels=(1, 5, 10, 20), window_hours=25,
                       label="tidal-mean salinity (psu)", title=None, merge_km=2.0, file_name=None):
    """A variable in every section against time (distance to the mouth × date), with contour lines.

    With the salinity it shows how far the salt intrudes along the estuary.

    Args:
        ds (xr.Dataset): Output read with :func:`~environmentaltools.models.barrier.read_output`.
        variable (str): Variable with dimensions ``(time, x)``.
        points (pd.DataFrame | None): Points marked as dashed lines and y-axis labels
            (from :func:`~environmentaltools.models.barrier.extract_points`).
        levels (list[float] | None): Contour lines (also marked on the colour bar).
        window_hours (float | None): Tidal mean before drawing; ``None`` draws the raw output.
        label (str): Colour bar label.
        title (str | None): Figure title.
        merge_km (float): Points closer than this share one label.
        file_name (str | Path | None): Save the figure here (and close it).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    dates, km = ds["date"].values, ds["km"].values
    values = ds[variable].values
    if window_hours:
        values = tidal_mean(values, window_hours, time_step_hours(ds))
    fig, ax = plt.subplots(figsize=(14, 6.5))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    mesh = ax.pcolormesh(dates, km, values.T, cmap=cmap, shading="auto",
                         vmin=np.nanmin(values), vmax=np.nanmax(values))
    bar = fig.colorbar(mesh, ax=ax, pad=0.01)
    if levels:
        contours = ax.contour(dates, km, values.T, levels=list(levels), colors=INK, linewidths=0.7)
        bar.add_lines(contours)
        label = f"{label} · lines: {', '.join(map(str, levels))}"
    if points is not None:
        for k in points["km_section"]:
            ax.axhline(k, color=SURFACE, lw=0.6, ls=(0, (4, 3)))
        ticks, labels = _merged_labels(points, merge_km)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_ylabel("distance to the mouth (km)", color=INK)
    ax.set_ylim(km.max(), 0)   # mouth on top
    ax.tick_params(colors=MUTED, labelsize=8)
    bar.set_label(label, color=INK)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=13, loc="left")
    fig.tight_layout()
    _save(fig, file_name)
    return fig


def plot_comparison(ref, new, stations_km=(0, 25, 50, 85), zoom_days=None, labels=("reference", "new"),
                    window_hours=25, title=None, file_name=None):
    """Two runs at a few stations: water level over a few days and salinity (tidal mean) over the whole run.

    Args:
        ref (xr.Dataset): Reference output.
        new (xr.Dataset): Output compared with it.
        stations_km (list[float]): Distances to the mouth of the stations (one column each).
        zoom_days (tuple[float, float] | None): Days since the start shown in the level row
            (default: the first four days).
        labels (tuple[str, str]): Legend of ``ref`` and ``new``.
        window_hours (float | None): Window of the tidal mean of the salinity; ``None`` draws the raw output.
        title (str | None): Figure title.
        file_name (str | Path | None): Save the figure here (and close it).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    n = min(ref.sizes["time"], new.sizes["time"])
    idx = nearest_sections(ref, stations_km)
    days = (ref["time"].values[:n] - ref["time"].values[0]) / 86400.0
    zoom_days = zoom_days or (days[0], min(days[0] + 4, days[-1]))
    zoom = (days >= zoom_days[0]) & (days <= zoom_days[1])
    step = time_step_hours(ref)
    start = pd.Timestamp(ref["date"].values[0])
    fig, axes = plt.subplots(2, len(idx), figsize=(4 * len(idx), 7.5), sharey="row", squeeze=False)
    for col, i in enumerate(idx):
        for ds, color, ls, name in ((ref, REF_COLOR, "-", labels[0]), (new, NEW_COLOR, "--", labels[1])):
            eta = ds["eta"].isel(x=i, time=slice(0, n)).values
            sal = ds["S"].isel(x=i, time=slice(0, n)).values
            if window_hours:
                sal = tidal_mean(sal, window_hours, step)
            axes[0, col].plot(days[zoom], eta[zoom], color=color, lw=2, ls=ls, label=name)
            axes[1, col].plot(days, sal, color=color, lw=2, ls=ls, label=name)
        axes[0, col].set_title(f"km {ref['km'].values[i]:.0f} from the mouth", color=INK, fontsize=11)
        axes[0, col].set_xlim(*zoom_days)
        for ax in axes[:, col]:
            _style(ax)
            ax.set_xlabel(f"days since {start:%Y-%m-%d}", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel(f"water level (m)\ndays {zoom_days[0]:g}-{zoom_days[1]:g}", color=INK)
    axes[1, 0].set_ylabel(f"tidal-mean salinity (psu)\n{window_hours:g}-h moving mean" if window_hours
                          else "salinity (psu)", color=INK)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=10, loc="upper right", ncol=2)
    if title:
        fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, file_name)
    return fig


def plot_section_parameters(along, columns=("nmann", "kh", "Sf", "beta"), labels=None, file_name=None):
    """Along-channel parameters of a case (Manning n, Kh, storage factor, β…) against the distance to the mouth.

    Args:
        along (pd.DataFrame): Along-channel table (``read_csv`` of ``along_channel_data``),
            or ``{label: table}`` to compare several cases.
        columns (list[str]): Columns drawn (one panel each; missing ones are skipped).
        labels (dict | None): Axis label of each column.
        file_name (str | Path | None): Save the figure here (and close it).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    tables = along if isinstance(along, dict) else {"": along}
    first = next(iter(tables.values()))
    columns = [c for c in columns if c in first]
    labels = {"nmann": "Manning n (s/m^1/3)", "kh": "Kh (m²/s)", "Sf": "storage factor Sf", "beta": "β",
              **(labels or {})}
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.2 * len(columns) + 0.6), sharex=True,
                             squeeze=False)
    for ax, col in zip(axes[:, 0], columns):
        for name, t in tables.items():
            km = (t["x"].max() - t["x"]) / 1000.0
            ax.plot(km, t[col], lw=1.6, label=name or None)
        ax.set_ylabel(labels.get(col, col), color=INK)
        _style(ax)
    axes[-1, 0].set_xlabel("distance to the mouth (km)", color=INK)
    axes[-1, 0].invert_xaxis()   # upstream on the left, mouth on the right
    if len(tables) > 1:
        axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, file_name)
    return fig
