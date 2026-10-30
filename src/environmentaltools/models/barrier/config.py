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
        if not d["variants"]:
            raise ValueError("The project has no 'cases' (use 'reference: {}' for the base case alone)")
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
