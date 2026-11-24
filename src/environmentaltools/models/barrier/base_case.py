"""Starter base case of BARRIER: a commented ``config.yaml`` with safe settings and the boundary series.

The geometry is separate (:func:`~environmentaltools.models.barrier.build_geometry`,
:func:`~environmentaltools.models.barrier.prismatic_geometry` or the step ``geometry``); together they make a case
that runs::

    barrier.prismatic_geometry(20000, dx=500, bottom_width=300, bed=-5, top_level=5, out_dir="base_case")
    barrier.new_base_case("base_case", "2020-01-01", "2020-01-05", tide={"amplitude": 1.0}, discharge=50)
"""

from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import yaml

from .config import check_config, input_files, read_config
from .inputs import write_series

#: Commented ``config.yaml`` written by :func:`new_base_case`.
CONFIG_TEMPLATE = Path(__file__).with_name("config_template.yaml")
#: Output variables of a new base case (salinity ``S`` is added when it is simulated). The bank coordinates give
#: the width ``W`` and the position of the points (:func:`~environmentaltools.models.barrier.read_output`).
OUTPUT_VARIABLES = ["eta", "level", "Q", "U", "A", "xl_utm_x", "xl_utm_y", "xr_utm_x", "xr_utm_y"]


def harmonic_tide(start, end, amplitude, period_hours=12.42, mean=0.0, phase_deg=0.0, step_minutes=10):
    """Tide of one constituent, ``mean + amplitude · sin(2π t / period + phase)``, from ``start`` to ``end``.

    It starts at the mean level and rising (``phase_deg = 0``): BARRIER starts with the water at 0 m
    (``initial_conditions: calm``), so a tide starting far from it makes a jump at the mouth.

    Args:
        start, end (str | datetime): Period.
        amplitude (float): Amplitude (m).
        period_hours (float): Period (h); 12.42 is M2.
        mean (float): Mean level (m).
        phase_deg (float): Phase (degrees).
        step_minutes (float): Time step (minutes).

    Returns:
        pd.Series: Levels indexed by date.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = pd.date_range(start, end, freq=pd.Timedelta(minutes=step_minutes))
    if dates[-1] < end:
        dates = dates.append(pd.DatetimeIndex([end]))
    hours = (dates - start).total_seconds().to_numpy() / 3600.0
    return pd.Series(mean + amplitude * np.sin(2 * np.pi * hours / period_hours + np.radians(phase_deg)),
                     index=dates, name="level")


def _series(value, start, end, tide=False):
    """A boundary series from a Series, a constant or (tide) the arguments of :func:`harmonic_tide`."""
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, dict):
        if not tide:
            raise ValueError("The discharge is a Series or a number")
        return harmonic_tide(start, end, **value)
    return pd.Series(float(value), index=pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end)]))


def _covers(series, start, end):
    """True if ``series`` (by date, or by seconds since ``start``) covers the whole period."""
    s = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(s.index):
        return s.index.min() <= start and s.index.max() >= end
    return s.index.min() <= 0 and s.index.max() >= (end - start).total_seconds()


def new_base_case(case_dir, start, end, tide, discharge, salinity=36.0, kh=100.0, timestep=3600,
                  courant_number=0.3, output_variables=None, name=None, config=None, overwrite=False, verbose=True):
    """Write a base case of BARRIER: a commented ``config.yaml`` with safe settings and the boundary series.

    The ``config.yaml`` (from ``config_template.yaml``) has what BARRIER needs to run and does not check: the
    sections ``initial_conditions`` (calm) and ``forcing``, the TVD limiter enabled, wetting and drying, the water
    level at the mouth (``tides.csv``) and the discharge at the upstream boundary (``discharge.csv``) and, with
    ``salinity``, sea water at the mouth and fresh water upstream. The geometry files are not written (see
    :func:`~environmentaltools.models.barrier.build_geometry`); their names are the usual ones.

    BARRIER starts with the water at rest at 0 m and, with salinity, fresh everywhere: leave a spin-up period
    before the period of analysis (days for the tide, weeks or more for the salt).

    Args:
        case_dir (str | Path): Case folder (created if needed; it may already hold the geometry).
        start, end (str | datetime): Period of the run.
        tide (pd.Series | float | dict): Water level at the mouth (m, same datum as the geometry): a series by date
            or by seconds since ``start``, a constant, or the arguments of :func:`harmonic_tide` (e.g.
            ``{"amplitude": 1.0}``).
        discharge (pd.Series | float): River discharge at the upstream boundary (m³/s): a series or a constant.
        salinity (float | None): Salinity of the sea (psu); ``None``: no salinity (faster, e.g. to calibrate levels).
        kh (float): Longitudinal dispersion (m²/s) where ``along_channel_data.csv`` has no ``kh`` column.
        timestep (float): Output interval (s).
        courant_number (float): Courant number of the run.
        output_variables (list[str] | None): Variables written (default :data:`OUTPUT_VARIABLES`, plus ``S``).
        name (str | None): Name of the run (default: the folder name).
        config (dict | None): Other options of BARRIER, merged into the ``config.yaml`` (same structure, e.g.
            ``{"hydrodynamics": {"manning_eta": True}}``). The file is then written without the comments of the
            template (``config_template.yaml`` keeps them).
        overwrite (bool): Replace an existing ``config.yaml`` and series.

    Returns:
        Path: The case folder.
    """
    case_dir = Path(case_dir)
    if (case_dir / "config.yaml").exists() and not overwrite:
        raise FileExistsError(f"{case_dir / 'config.yaml'} exists (overwrite=True to replace it)")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError("'end' must be after 'start'")
    if output_variables:
        variables = list(output_variables)
    else:
        variables = OUTPUT_VARIABLES + (["S"] if salinity is not None else [])
    text = Template(CONFIG_TEMPLATE.read_text(encoding="utf-8")).substitute(
        name=name or case_dir.name, start_date=f"{start:%Y-%m-%dT%H:%M:%S}", end_date=f"{end:%Y-%m-%dT%H:%M:%S}",
        timestep=f"{timestep:g}", output_variables="[" + ", ".join(variables) + "]",
        discharge_file="discharge.csv", tide_file="tides.csv", courant_number=f"{courant_number:g}",
        salinity_enabled="true" if salinity is not None else "false", kh=f"{kh:g}",
        ocean_salinity=f"{36.0 if salinity is None else salinity:g}")
    problems = check_config(yaml.safe_load(text))
    if problems:                                       # the template must never give one
        raise RuntimeError(f"config_template.yaml: {problems}")
    if config:
        from ..swan.config import deep_merge

        merged = deep_merge(yaml.safe_load(text), config)
        for msg in check_config(merged):
            print(f"Warning: {msg} (set in 'config')")
        text = ("# BARRIER configuration of a base case, written by environmentaltools.models.barrier.new_base_case\n"
                "# with extra options. Every default option, commented: environmentaltools/models/barrier/"
                "config_template.yaml\n" + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "config.yaml").write_text(text, encoding="utf-8")
    for value, file_name, names, is_tide in ((tide, "tides.csv", ("time_seconds", "level_m"), True),
                                             (discharge, "discharge.csv", ("time_seconds", "discharge_m3s"), False)):
        series = _series(value, start, end, tide=is_tide)
        if not _covers(series, start, end):
            print(f"Warning: {file_name} does not cover the whole run ({start} to {end})")
        write_series(series, case_dir / file_name, start=start, names=names)

    missing = [k for k, f in input_files(case_dir, read_config(case_dir)).items() if not f.exists()]
    if verbose:
        print(f"Base case in {case_dir}: config.yaml, tides.csv, discharge.csv"
              + (f"; still missing: {', '.join(missing)} (build_geometry, prismatic_geometry or the step "
                 "'geometry')" if missing else ""))
    return case_dir


def read_series(file_name):
    """A boundary series from a csv file: first column the time (dates, or seconds since the start of the run),
    second column the value. BARRIER files (header line starting with ``;``) are read too.

    Returns:
        pd.Series: Values indexed by date or by seconds.
    """
    from .inputs import read_csv

    import re

    f = Path(file_name)
    with open(f) as fh:
        first = fh.readline().strip()
    if first.startswith(";"):                               # BARRIER file
        df = read_csv(f)
    else:
        sep = ";" if ";" in first else "," if "," in first else r"\s+"
        field = re.split(sep, first)[0].strip()
        is_value = pd.notna(pd.to_numeric(field, errors="coerce")) or pd.notna(pd.to_datetime(field, errors="coerce"))
        df = pd.read_csv(f, sep=sep, engine="python", header=None if is_value else 0)   # header: not a time
    if df.shape[1] < 2:
        raise ValueError(f"{f.name}: needs two columns (time, value)")
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    index = t.to_numpy(dtype=float) if t.notna().all() else pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
    return pd.Series(pd.to_numeric(df.iloc[:, 1]).to_numpy(dtype=float), index=index, name=df.columns[1])


#: Keys of the ``simulation`` section of a project file (arguments of :func:`new_base_case`).
SIMULATION_KEYS = {"start", "end", "tide", "discharge", "salinity", "kh", "timestep", "courant_number",
                   "output_variables", "name", "config"}
#: Files written by the ``simulation`` section in the base case.
SIMULATION_FILES = ("config.yaml", "tides.csv", "discharge.csv")


def simulation_settings(project):
    """Arguments of :func:`new_base_case` from the ``simulation`` section of a project file (series files read).

    Returns:
        tuple[dict, dict]: The arguments and ``{key: Path}`` of the series files.

    Raises:
        ValueError: If a key is unknown or ``start``, ``end``, ``tide`` or ``discharge`` is missing.
    """
    cfg = dict(project["simulation"] or {})
    unknown = set(cfg) - SIMULATION_KEYS
    if unknown:
        raise ValueError(f"simulation: unknown keys {sorted(unknown)}; use {sorted(SIMULATION_KEYS)}")
    missing = {"start", "end", "tide", "discharge"} - set(cfg)
    if missing:
        raise ValueError(f"simulation: missing {sorted(missing)}")
    files = {}
    for k in ("tide", "discharge"):
        if isinstance(cfg[k], str):
            files[k] = project.path(cfg[k])
            cfg[k] = read_series(files[k])
    return cfg, files


def prepare_simulation(project, force=False, verbose=True):
    """Write ``config.yaml`` and the boundary series of the base case from the ``simulation`` section, if pending.

    The section holds the arguments of :func:`new_base_case`, with the series as files (relative to the project
    file; see :func:`read_series`), numbers or, for the tide, the arguments of :func:`harmonic_tide`::

        simulation:
          start: 2008-07-01
          end: 2008-11-30
          tide: inputs/tide.csv
          discharge: inputs/discharge.csv
          salinity: 36.5
          config: {hydrodynamics: {manning_eta: true}}

    The files are written again when the section or a series file (size and date) changes, or when they are
    missing or were edited by hand (a copy of the edited file is kept in ``base_report_dir/edited``): the section
    is the place to change them. Since the base case changes, :func:`~environmentaltools.models.barrier.prepare`
    then creates the cases again.

    Returns:
        bool: True if the files were written.
    """
    if not project["simulation"]:
        return False
    from .geometry import pending

    kwargs, files = simulation_settings(project)
    base, report = project.base_case, project.path(project["base_report_dir"])
    why = "forced" if force else pending(report / "simulation_signature.json", project["simulation"], files, base,
                                         SIMULATION_FILES, report)
    if why is None:
        if verbose:
            print(f"Simulation: up to date ({base})")
        return False
    if verbose:
        print(f"Simulation: {why} → written again")
    new_base_case(base, overwrite=True, verbose=False, **kwargs)
    from .geometry import save_signature

    save_signature(report / "simulation_signature.json", project["simulation"], files, base, SIMULATION_FILES)
    return True
