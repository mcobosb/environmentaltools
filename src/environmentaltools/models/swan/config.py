"""Configuration of a SWAN project (YAML file + Python overrides).

A project is described by a YAML file (see ``swan_template.yaml`` in this
folder). Relative paths are resolved against the folder of the YAML file.
Every value can be overridden from Python with a nested dictionary.

Physics and numerics are dictionaries of *named SWAN commands* (written
verbatim, in order), so that a variant can replace or remove (``null``) one of
them, e.g.::

    variants:
      S0: {physics: {wcap: "WCAP KOMEN delta=1"}}
      D2: {boundary_spectrum: {spreading_value: 2.0}}

The variants can also be a table (``variants_file``, Excel or csv): one row per
variant (column ``variant``) and one column per changed key written with dots
(``physics.wcap``, ``boundary_spectrum.spreading_value``…); an empty cell keeps the
base value and ``off`` removes a command. The cases are every sea state of the
states table (``states: file``) × every variant.
"""

import copy
from pathlib import Path

import yaml

from ..common.grids import Grid

#: Defaults: stationary SWAN 41.51 set-up validated against Delft3D-WAVE 4.07 in the
#: ZyS article (Bergillos et al., 2017 + Komen delta = 0).
DEFAULTS = {
    "project": "swan",
    "cases_dir": "cases",
    "results_dir": "results",
    "archive_dir": None,
    "run_dir": None,
    "executable": "auto",
    "search_dirs": ["C:/Program Files/Deltares", "C:/Program Files (x86)/Deltares", "C:/SWAN", "/usr/local/bin"],
    "threads": None,
    "timeout_hours": 3,
    "keep_run_dir": False,
    "grids": {},
    "spectral_grid": {"ndir": 36, "fmin": 0.05, "fmax": 1.0, "msc": 24},
    "set": {"level": 0.0, "nor": 90.0, "depmin": 0.05, "maxmes": 1000, "maxerr": 2,
            "grav": 9.81, "rho": 1025.0, "inrhog": 1, "convention": "NAUTICAL"},
    "boundary_spectrum": {"shape": "JONSWAP", "gamma": 3.3, "period": "PEAK",
                          "spreading": "POWER", "spreading_value": 4.0},
    "physics": {
        "gen3": "GEN3 KOMEN DRAG WU",
        "wcap": "WCAP KOMEN delta=0",
        "breaking": "BREAK CON 1.00 0.73",
        "friction": "FRIC JON 0.067",
        "limiter": "LIM 10 1",
        "setup": "SETUP",
    },
    "numerics": {
        "dir": "NUM DIR cdd=0.50 SIGIM css=0.50",
        "accur": "NUM ACCUR 0.02 0.02 0.02 97.0 50",
    },
    "outputs": {
        "points": None,
        "point_quantities": ["XP", "YP", "DEPTH", "HSIGN", "TPS", "DIR", "DSPR", "WIND"],
        "point_spectra": False,
        "maps": {},
        "figures": None,
    },
    "states": {
        "file": None,
        "sheet": None,
        "id": None,
        "columns": {"Hs": "Hs", "Tp": "Tp", "Dir": "DirM", "Vv": "Vv", "DirV": "DirV", "spreading": None,
                    "level": None},
        "min_wind": 0.01,
    },
    "variants": {},
    "variants_file": None,
    "overview_file": None,
    "postprocess_dir": "postprocess",
    "postprocess": {
        "maps": {"cases": [], "quantity": None},      # map figures of these cases (all their saved maps)
    },
}

#: Text of a cell of the variants table that removes a SWAN command (``null`` in the YAML).
REMOVE = {"off", "null", "none", "remove"}


def read_variants_table(file_name, sheet_name=None):
    """Read the variants table (xlsx or csv): one row per variant, one column per changed key.

    Columns: ``variant`` (name; the cases are ``<state>_<variant>``), ``description`` or columns
    starting with ``_`` (free text) and any key of the project file written with dots, e.g.
    ``physics.wcap`` = ``WCAP KOMEN delta=1``, ``boundary_spectrum.spreading_value`` = 2,
    ``numerics.accur``. An empty cell keeps the base value; ``off`` removes a command.

    Returns:
        dict: ``{variant: nested dict of changes}``, as ``variants:`` in the project file.
    """
    from ..common.tables import read_changes_table, set_nested

    def column(col, value, changes, f):
        if value is False or (isinstance(value, str) and value.strip().lower() in REMOVE):   # YAML reads off as False
            set_nested(changes, col.split("."), None)
            return True
        return False

    return read_changes_table(file_name, "variant", sheet_name, "variants", column)


def deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``.

    ``None`` disables a command of a command dict; an empty dict replaces the whole entry.
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and v and isinstance(out.get(k), dict):     # {} replaces (e.g. variants: {})
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class SwanConfig:
    """SWAN project configuration.

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
            self.file, raw, self.base_dir = None, dict(source), Path.cwd()
        self.data = deep_merge(deep_merge(DEFAULTS, raw), overrides)
        self.data["variants"] = {k: (v or {}) for k, v in (self.data["variants"] or {}).items()}
        f = self.variants_file
        if f is not None and f.exists():
            for name, v in read_variants_table(f).items():
                if name in self.data["variants"]:
                    raise ValueError(f"Variant '{name}' is both in the project file and in {f.name}")
                self.data["variants"][name] = v
        self._grids = None
        self.validate()

    # -------------------------------------------------------------- access
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
    def variants_file(self):
        """Table of variants (xlsx/csv), if the project uses one (see :func:`read_variants_table`)."""
        return self.path(self.data["variants_file"])

    @property
    def overview_file(self):
        """Excel file to review the cases before running (default ``<cases_dir>/cases_overview.xlsx``)."""
        return self.path(self.data["overview_file"]) if self.data["overview_file"] else \
            self.cases_dir / "cases_overview.xlsx"

    @property
    def postprocess_dir(self):
        return self.path(self.data["postprocess_dir"])

    @property
    def cases_dir(self):
        return self.path(self.data["cases_dir"])

    @property
    def results_dir(self):
        return self.path(self.data["results_dir"])

    @property
    def archive_dir(self):
        """Folder for heavy outputs kept only on the computing machine (maps with ``store: archive``)."""
        return self.path(self.data["archive_dir"])

    def map_store(self, grid):
        """Where the map of ``grid`` is kept: ``"results"`` (default) or ``"archive"``."""
        return ((self.data["outputs"].get("maps") or {}).get(grid) or {}).get("store", "results")

    @property
    def grid_names(self):
        """Grid names in computation order (parents before nested grids)."""
        names, pending = [], list(self.data["grids"])
        while pending:
            for g in list(pending):
                parent = self.data["grids"][g].get("nested_in")
                if parent is None or parent in names:
                    names.append(g)
                    pending.remove(g)
                    break
            else:
                raise ValueError(f"Nesting cycle or unknown parent among grids {pending}")
        return names

    def children(self, grid_name):
        """Grids nested directly in ``grid_name``."""
        return [g for g in self.grid_names if self.data["grids"][g].get("nested_in") == grid_name]

    def grids(self):
        """Load the grids (cached) as :class:`Grid` objects."""
        if self._grids is None:
            self._grids = {name: self.load_grid(name) for name in self.grid_names}
        return self._grids

    def load_grid(self, name):
        """Grid ``name``: Delft3D ``grd`` (+ ``dep``) or ``regular``; ``bathymetry`` (xyz file)
        interpolates the depth onto the nodes when there is no ``dep``."""
        g = self.data["grids"][name]
        if "grd" in g:
            grid = Grid.from_delft(self.path(g["grd"]), self.path(g.get("dep")), name)
        else:
            r = g["regular"]
            grid = Grid.regular_grid(r["xp"], r["yp"], r.get("alp", 0.0), r["dx"], r["dy"], r["nx"], r["ny"],
                                     name, g.get("spherical", False))
        if "bathymetry" in g and grid.depth is None:
            import pandas as pd

            b = g["bathymetry"]
            xyz = pd.read_csv(self.path(b["file"]), sep=b.get("sep", r"[\s,;]+"), engine="python",
                              header=b.get("header", None), comment=b.get("comment", "#"))
            cols = b.get("columns", [0, 1, 2])
            pick = (lambda c: xyz[c]) if isinstance(cols[0], str) else (lambda c: xyz.iloc[:, c])
            grid.set_depth_from_points(pick(cols[0]), pick(cols[1]), pick(cols[2]),
                                       b.get("method", "linear"), b.get("positive_down", True))
        if grid.depth is None:
            raise ValueError(f"Grid '{name}' has no depth: give 'dep' or 'bathymetry'")
        return grid

    def variant(self, name):
        """Configuration of a variant (``None`` → the base configuration)."""
        if not name:
            return self
        new = SwanConfig.__new__(SwanConfig)
        new.file, new.base_dir, new._grids = self.file, self.base_dir, self._grids
        new.data = deep_merge(self.data, self.data["variants"][name])
        return new

    # ---------------------------------------------------------- validation
    def validate(self):
        d = self.data
        if not d["grids"]:
            raise ValueError("The configuration has no grids")
        for name, g in d["grids"].items():
            if "grd" not in g and "regular" not in g:
                raise ValueError(f"Grid '{name}': give 'grd' (+ 'dep') or 'regular'")
            if g.get("nested_in") and g["nested_in"] not in d["grids"]:
                raise ValueError(f"Grid '{name}': unknown parent '{g['nested_in']}'")
            if not g.get("nested_in") and not g.get("boundaries"):
                raise ValueError(f"Grid '{name}' is not nested and has no 'boundaries'")
        self.grid_names  # checks cycles
        for g, m in (d["outputs"].get("maps") or {}).items():
            if g not in d["grids"]:
                raise ValueError(f"Map for unknown grid '{g}'")
            if m.get("store", "results") not in ("results", "archive"):
                raise ValueError(f"Map '{g}': store must be 'results' or 'archive'")
            if m.get("store") == "archive" and not d.get("archive_dir"):
                raise ValueError(f"Map '{g}' has store: archive but there is no 'archive_dir'")
        for v in d["variants"]:
            if "grids" in d["variants"][v]:
                raise ValueError(f"Variant '{v}': changing grids is not supported; use another project")
        return True


def load_config(source, overrides=None):
    """Load a SWAN project configuration (YAML file or dict) with optional overrides."""
    return SwanConfig(source, overrides)
