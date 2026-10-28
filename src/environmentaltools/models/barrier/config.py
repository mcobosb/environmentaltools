"""Configuration of BARRIER cases (the model's own ``config.yaml``) and projects (``barrier.yaml``).

A *case* is a folder with the ``config.yaml`` read by BARRIER and the input files it
names (relative to that folder). A *project* (optional) is a YAML file with the
executable, a base case and named variants, so that a batch of cases can be prepared
and run in one go::

    executable: null          # BARRIER executable (null → environment variable BARRIER_EXE or PATH)
    base_case: base           # case folder used as starting point
    cases_dir: cases          # one folder per variant
    results_dir: results      # one folder per variant with the NetCDF output
    variants:
      reference: {}
      rough:
        config: {hydrodynamics: {courant_number: 0.2}}
        along_channel: {nmann: {scale: 1.1}}
        files: {tides.csv: tides_2010.csv}

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
    "timeout_hours": None,
    "variants": {},
}

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
        self.data = deep_merge(deep_merge(DEFAULTS, raw), overrides)
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
    def case_names(self):
        return list(self.data["variants"])

    def validate(self):
        d = self.data
        if not d["base_case"]:
            raise ValueError("The project has no 'base_case'")
        if not d["variants"]:
            raise ValueError("The project has no 'variants' (use 'reference: {}' for the base case alone)")
        for name, v in d["variants"].items():
            unknown = set(v or {}) - {"config", "along_channel", "files"}
            if unknown:
                raise ValueError(f"Variant '{name}': unknown keys {sorted(unknown)} "
                                 "(use config, along_channel, files)")
        return True


def load_project(source, overrides=None):
    """Load a BARRIER project (YAML file or dict) with optional overrides."""
    return BarrierProject(source, overrides)
