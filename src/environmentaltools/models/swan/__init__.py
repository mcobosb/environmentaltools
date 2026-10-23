"""SWAN projects driven by a YAML file.

When to use: spectral wave propagation over real bathymetry in 2-D
(refraction, diffraction, nesting from deep water to the coast) for batches
of sea states. It gives wave fields and spectra, not currents or sediment
transport (see :mod:`environmentaltools.models.swan_copla`).

Typical use::

    from environmentaltools.models import swan

    cfg = swan.load_config("swan.yaml")
    swan.prepare(cfg)                 # grids, nesting points and one folder per case
    swan.run(cfg)                     # on the machine with SWAN (resumable)
    pts = swan.read_points(cfg)       # one row per case, grid and output point
    swan.summary(cfg)                 # convergence and errors

See ``swan_template.yaml`` in this folder for every option.
"""

from pathlib import Path

from .config import DEFAULTS, SwanConfig, deep_merge, load_config
from .inputs import build_input, write_grid_files, write_nest_points
from .cases import boundary_distances, case_names, load_states, prepare
from .run import run, swan_executable
from .read import (quantity_columns, read_map, read_points, read_print, read_spectra, read_spectra_case,
                   spectrum_parameters,
                   read_table, summary)
from .plot import plot_case_maps, plot_grids, plot_map, plot_wave_field

TEMPLATE = Path(__file__).with_name("swan_template.yaml")

__all__ = [
    "DEFAULTS", "SwanConfig", "deep_merge", "load_config", "TEMPLATE",
    "build_input", "write_grid_files", "write_nest_points",
    "boundary_distances", "case_names", "load_states", "prepare",
    "run", "swan_executable",
    "quantity_columns", "read_map", "read_points", "read_print", "read_spectra", "read_spectra_case", "spectrum_parameters",
    "read_table", "summary",
    "plot_case_maps", "plot_grids", "plot_map", "plot_wave_field",
]
