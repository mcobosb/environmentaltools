"""SWAN projects driven by a YAML file.

When to use: spectral wave propagation over real bathymetry in 2-D
(refraction, diffraction, nesting from deep water to the coast) for batches
of sea states. It gives wave fields and spectra, not currents or sediment
transport (see :mod:`environmentaltools.models.swan_copla`).

Workflow, from one project file (start from ``swan_template.yaml`` in this folder): the
base configuration (grids, physics, numerics, outputs), the sea states table (one row per
state, ``states: file``) and, optionally, the variants table (``variants_file``: one row per
variant, one column per changed key, e.g. ``physics.wcap``). The cases are states × variants::

    from environmentaltools.models import swan

    cfg = swan.load_config("swan.yaml")
    swan.prepare(cfg)          # grids, nesting points, one folder per case + overview to review them
    swan.run(cfg)              # on the machine with SWAN (resumable)
    swan.postprocess(cfg)      # points.xlsx, convergence.xlsx and maps of chosen cases

or from the terminal: ``python -m environmentaltools.models.swan all swan.yaml``.

Each step only does what is pending: a case whose state or variant changed (or every case,
if the grids changed) is written again and its result becomes pending; the rest are kept.
The results can also be read directly (``read_points``, ``summary``, ``read_map``…).

See ``swan_template.yaml`` in this folder for every option.
"""

from pathlib import Path

from .config import DEFAULTS, SwanConfig, deep_merge, load_config, read_variants_table
from .inputs import build_input, write_grid_files, write_nest_points
from .cases import boundary_distances, case_names, load_states, prepare, write_overview
from .run import run, swan_executable
from .read import (quantity_columns, read_map, read_points, read_print, read_spectra, read_spectra_case,
                   spectrum_parameters,
                   read_table, summary)
from .plot import inspect_grids, plot_case_maps, plot_grids, plot_map, plot_wave_field
from .postprocess import postprocess

TEMPLATE = Path(__file__).with_name("swan_template.yaml")

__all__ = [
    "DEFAULTS", "SwanConfig", "deep_merge", "load_config", "read_variants_table", "TEMPLATE",
    "build_input", "write_grid_files", "write_nest_points",
    "boundary_distances", "case_names", "load_states", "prepare", "write_overview", "postprocess",
    "run", "swan_executable",
    "quantity_columns", "read_map", "read_points", "read_print", "read_spectra", "read_spectra_case", "spectrum_parameters",
    "read_table", "summary",
    "inspect_grids", "plot_case_maps", "plot_grids", "plot_map", "plot_wave_field",
]
