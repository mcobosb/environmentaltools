"""Configuration of BARRIER cases (the model's own ``config.yaml``) and projects (``barrier.yaml``).

A *case* is a folder with the ``config.yaml`` read by BARRIER and the input files it
names (relative to that folder). A *project* is a YAML file with a base case, the list of
cases (the base case plus changes), where everything is written, the executable and the
postprocessing, so that the whole workflow runs from it (prepare → run → postprocess)::

    base_case: base           # case folder used as starting point
    executable: null          # BARRIER executable or list of candidates (null → BARRIER_EXE or PATH)
    cases:
      reference: {}
      rough:
        config: {hydrodynamics: {courant_number: 0.2}}
        along_channel: {nmann: {scale: 1.1}}
        files: {tides.csv: tides_2010.csv}
    postprocess:
      points: points.xlsx     # series at these points (columns name, km to the mouth)
      reference: {case: reference}

See ``barrier_template.yaml`` in this folder for every option.
"""

import copy
from pathlib import Path

import yaml

from ..swan.config import deep_merge

#: Defaults of a BARRIER project file.
DEFAULTS = {
    "executable": None,
    "base_case": None,
    "cases_dir": "cases",
    "results_dir": "results",
    "postprocess_dir": "postprocess",
    "overview_file": None,
    "timeout_hours": None,
    "work_dir": None,
    "cases_file": None,
    "variants": {},
    "postprocess": {
        "points": None,
        "points_columns": {"name": "name", "km": "km"},
        "variables": ["eta", "S"],
        "reference": None,
        "tolerances": None,
        "stations_km": None,
        "zoom_days": None,
        "window_hours": 25,
        "along_channel": {"variable": "S", "levels": [1, 5, 10, 20]},
        "legend_notes": {},
    },
}

#: Keys of a case in the project file.
CASE_KEYS = {"config", "along_channel", "files"}
#: Columns of the cases table that are not changes (free text).
NOTE_COLUMNS = {"description", "notes", "note", "comment", "comments"}


def _cell(v):
    """Value of a cell of the cases table: numbers as numbers (integers without decimals), text parsed as YAML
    (so ``true``, ``[[0, 10], [5000, 20]]`` or ``2010-01-01T00:00:00`` work), empty → ``None``."""
    import math

    if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    if hasattr(v, "isoformat"):                                  # dates typed as dates in Excel
        return v.isoformat()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    if hasattr(v, "item"):                                       # numpy scalars
        return _cell(v.item())
    text = str(v).strip()
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return parsed if isinstance(parsed, (int, float, bool, list, dict)) and not isinstance(parsed, str) else text


def _set(d, keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def read_cases_table(file_name, sheet_name=None):
    """Read the cases table (xlsx or csv): one row per case, one column per change of the base case.

    Columns:

    - ``case``: name of the case (also its folder name).
    - ``description`` (or ``notes``, ``comment``) and columns starting with ``_``: free text, ignored.
    - ``along_channel.<column>.<option>``: section parameters (``nmann``, ``kh``, ``Sf``, ``beta``) with the
      options of :func:`~environmentaltools.models.barrier.set_section_parameter`, e.g.
      ``along_channel.nmann.scale`` = 1.1, ``along_channel.kh.points`` = ``[[0, 10], [50000, 300]]``.
      ``along_channel.<column>.file`` reads the ``points`` from a csv (columns x, value) next to the table.
    - ``files.<name in the case>``: file that replaces that input (relative to the table), e.g.
      ``files.tides.csv`` = ``tides_2010.csv``.
    - Anything else is a key of BARRIER's ``config.yaml`` written with dots, e.g.
      ``hydrodynamics.courant_number`` or ``run.end_date`` (``config.`` in front is optional).

    An empty cell means "as in the base case".

    Args:
        file_name (str | Path): Excel (sheet ``cases``, else the first one) or csv file.
        sheet_name (str | None): Sheet to read.

    Returns:
        dict: ``{case: {"config": …, "along_channel": …, "files": …}}`` as in the project file.
    """
    import pandas as pd

    f = Path(file_name)
    if f.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.ExcelFile(f).sheet_names
        df = pd.read_excel(f, sheet_name=sheet_name or ("cases" if "cases" in sheets else sheets[0]))
    else:
        df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    if "case" not in df:
        raise ValueError(f"{f.name}: the cases table needs a 'case' column")
    cases = {}
    for _, row in df.iterrows():
        name = _cell(row["case"])
        if name is None:
            continue
        name = str(name)
        if name in cases:
            raise ValueError(f"{f.name}: case '{name}' appears twice")
        case = {}
        for col, v in row.items():
            value = _cell(v)
            if col == "case" or value is None or col.lower() in NOTE_COLUMNS or col.startswith("_"):
                continue
            if col.startswith("files."):
                _set(case, ["files", col[len("files."):]], str(value))
                continue
            keys = col.split(".")
            if keys[0] == "along_channel":
                if len(keys) != 3:
                    raise ValueError(f"{f.name}: column '{col}' must be along_channel.<column>.<option>")
                if keys[2] == "file":
                    pts = pd.read_csv(f.parent / str(value))
                    keys, value = keys[:2] + ["points"], pts.iloc[:, :2].to_numpy(dtype=float).tolist()
                _set(case, keys, value)
            else:
                _set(case, ["config", *(keys[1:] if keys[0] == "config" else keys)], value)
        cases[name] = case
    return cases

#: Keys of a case ``config.yaml`` that name input files (relative to the case folder).
FILE_KEYS = [
    ("geometry", "along_channel_file"),
    ("geometry", "cross_sections_file"),
    ("hydrodynamics", "boundary_conditions", "upstream", "file"),
    ("hydrodynamics", "boundary_conditions", "downstream", "file"),
    ("hydrodynamics", "forcing", "tributaries_file"),
    ("transport", "salinity", "initial_file"),
    ("transport", "salinity", "boundary_conditions", "downstream", "file"),
    ("transport", "temperature", "initial_file"),
    ("transport", "temperature", "boundary_conditions", "upstream", "file"),
    ("transport", "temperature", "boundary_conditions", "downstream", "file"),
    ("transport", "temperature", "heat_flux", "heat_flux_file"),
    ("transport", "sediment", "properties_file"),
]


def read_config(case_dir):
    """Read the ``config.yaml`` of a case folder as a dict."""
    return yaml.safe_load((Path(case_dir) / "config.yaml").read_text(encoding="utf-8")) or {}


def write_config(config, case_dir):
    """Write ``config`` as the ``config.yaml`` of a case folder (comments of the original are lost)."""
    f = Path(case_dir) / "config.yaml"
    f.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return f


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def input_files(case_dir, config=None):
    """Input files named by the ``config.yaml`` of a case.

    Args:
        case_dir (str | Path): Case folder.
        config (dict | None): Configuration (read from the folder if not given).

    Returns:
        dict: ``{"geometry.along_channel_file": Path, …}`` for every non-empty file key.
    """
    config = config if config is not None else read_config(case_dir)
    out = {}
    for keys in FILE_KEYS:
        v = _get(config, keys)
        if v not in (None, ""):
            out[".".join(keys)] = Path(case_dir) / str(v)
    return out


def start_date(config):
    """Start of the simulation (``run: start_date``) as a Timestamp."""
    import pandas as pd

    return pd.Timestamp(config["run"]["start_date"])


class BarrierProject:
    """BARRIER project configuration (``barrier.yaml``).

    Args:
        source (str | Path | dict): YAML file or dictionary.
        overrides (dict | None): Values that replace those of ``source``.

    Attributes:
        data (dict): Full configuration (defaults + file + overrides).
        base_dir (Path): Folder used to resolve relative paths.
    """

    def __init__(self, source, overrides=None):
        if isinstance(source, (str, Path)):
            self.file = Path(source).resolve()
            raw = yaml.safe_load(self.file.read_text(encoding="utf-8")) or {}
            self.base_dir = self.file.parent
        else:
            self.file, raw, self.base_dir = None, copy.deepcopy(dict(source)), Path.cwd()
        if "cases" in raw:                       # 'cases' is the documented name, 'variants' the old one
            raw["variants"] = raw.pop("cases")
        self.data = deep_merge(deep_merge(DEFAULTS, raw), overrides)
        self.data["variants"] = {name: (v or {}) for name, v in (self.data["variants"] or {}).items()}
        f = self.cases_file
        if f is not None and f.exists():
            # files: of the table are relative to the table: make them relative to the project file, so that
            # the case (and its fingerprint) is the same on every computer (Mac, WSL…)
            import os

            for name, case in read_cases_table(f).items():
                if name in self.data["variants"]:
                    raise ValueError(f"Case '{name}' is both in the project file and in {f.name}")
                for k, v in (case.get("files") or {}).items():
                    case["files"][k] = Path(os.path.relpath(f.parent / v, self.base_dir)).as_posix()
                self.data["variants"][name] = case
        self.validate()

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def path(self, value):
        """Resolve a path relative to the YAML folder."""
        if value is None:
            return None
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def base_case(self):
        return self.path(self.data["base_case"])

    @property
    def cases_dir(self):
        return self.path(self.data["cases_dir"])

    @property
    def results_dir(self):
        return self.path(self.data["results_dir"])

    @property
    def cases_file(self):
        """Table of cases (xlsx/csv), if the project uses one (see :func:`read_cases_table`)."""
        return self.path(self.data["cases_file"]) if self.data["cases_file"] else None

    @property
    def postprocess_dir(self):
        return self.path(self.data["postprocess_dir"])

    @property
    def overview_file(self):
        """Excel file comparing the base case and the cases (default ``<cases_dir>/cases_overview.xlsx``)."""
        return self.path(self.data["overview_file"]) if self.data["overview_file"] else \
            self.cases_dir / "cases_overview.xlsx"

    @property
    def work_dir(self):
        """Local folder where BARRIER writes while running (``true`` → the system temporary folder)."""
        w = self.data["work_dir"]
        if w is True:
            import tempfile

            return Path(tempfile.gettempdir())
        return self.path(w) if w else None

    @property
    def executable(self):
        """Configured executable(s): one path or a list of candidates (the first that exists is used)."""
        exe = self.data["executable"]
        if isinstance(exe, (list, tuple)):
            return [self.path(e) for e in exe]
        return self.path(exe)

    @property
    def case_names(self):
        return list(self.data["variants"])

    def validate(self):
        d = self.data
        if not d["base_case"]:
            raise ValueError("The project has no 'base_case'")
        if not d["variants"] and not d["cases_file"]:
            raise ValueError("The project has no 'cases' nor 'cases_file' (use 'reference: {}' for the base case alone)")
        for name, v in d["variants"].items():
            unknown = set(v) - CASE_KEYS
            if unknown:
                raise ValueError(f"Case '{name}': unknown keys {sorted(unknown)} (use config, along_channel, files)")
        ref = d["postprocess"].get("reference") or {}
        if ref and not (("case" in ref) ^ ("file" in ref)):
            raise ValueError("postprocess: reference needs either 'case' (a case of the list) or 'file' (a NetCDF)")
        if ref.get("case") and ref["case"] not in d["variants"]:
            raise ValueError(f"postprocess: reference case '{ref['case']}' is not in the list of cases")
        return True


def load_project(source, overrides=None):
    """Load a BARRIER project (YAML file or dict) with optional overrides."""
    return BarrierProject(source, overrides)
