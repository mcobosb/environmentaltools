"""BARRIER 1-D estuarine model: case set-up, execution, results and figures.

BARRIER (https://github.com/mcobosb/Barrier, M. Cobos) solves the cross-section
averaged Saint-Venant equations with salinity (and temperature and sediment)
transport along a channel described by cross-section tables. Section ``x = 0``
is the upstream boundary and the last section is the mouth.

When to use: tidal propagation and salt intrusion along an estuary or a tidal
river, over weeks to years, where the cross-section average is enough. It does
not resolve lateral or vertical structure (stratification, secondary currents).

Not covered yet: building ``cross-sections.csv`` and ``along_channel_data.csv``
from a channel axis and a DEM (to be added; for now they come from an existing case).

A *case* is a folder with BARRIER's ``config.yaml`` and the CSV files it names.
Typical use::

    from environmentaltools.models import barrier

    # one case from another, changing the config and the along-channel parameters
    barrier.create_case("run_002_rough", "run_001_reference",
                        config={"hydrodynamics": {"courant_number": 0.2}},
                        along_channel={"nmann": {"scale": 1.1}})
    nc = barrier.run_case("run_002_rough", "outputs/run_002_rough")

    ds = barrier.read_output(nc)                       # (time, x) with 'date' and 'km' coordinates
    points, series = barrier.extract_points(ds, {"Mouth": 0, "Upstream gauge": 80})
    barrier.plot_point_series(series["S"], "salinity (psu)", file_name="salinity.png")
    barrier.plot_along_channel(ds, "S", points, file_name="salt_intrusion.png")

Or a batch of variants from a project file (see ``barrier_template.yaml`` in this folder)::

    project = barrier.load_project("barrier.yaml")
    barrier.prepare(project)              # one case folder per variant
    barrier.run(project)                  # resumable
    outputs = barrier.read_project(project)
"""

from pathlib import Path

from .config import BarrierProject, input_files, load_project, read_config, start_date, write_config
from .inputs import (CROSS_SECTION_COLUMNS, MONOTONIC, check_cross_sections, fix_cross_sections, read_csv,
                     section_profile, set_section_parameter, write_csv, write_series, write_tributaries)
from .cases import create_case, prepare
from .run import barrier_executable, has_time_steps, is_readable, run, run_case
from .read import (COMPARE_TOLERANCES, compare, distance_to_mouth, extract_points, latest_output,
                   nearest_sections, read_output, read_project, tidal_mean, time_step_hours)
from .plot import plot_along_channel, plot_comparison, plot_point_series, plot_section_parameters

TEMPLATE = Path(__file__).with_name("barrier_template.yaml")

__all__ = [
    "BarrierProject", "load_project", "read_config", "write_config", "input_files", "start_date", "TEMPLATE",
    "CROSS_SECTION_COLUMNS", "MONOTONIC", "read_csv", "write_csv", "write_series", "write_tributaries",
    "section_profile", "set_section_parameter", "check_cross_sections", "fix_cross_sections",
    "create_case", "prepare",
    "barrier_executable", "has_time_steps", "is_readable", "run_case", "run",
    "COMPARE_TOLERANCES", "read_output", "latest_output", "distance_to_mouth", "time_step_hours", "tidal_mean",
    "nearest_sections", "extract_points", "compare", "read_project",
    "plot_point_series", "plot_along_channel", "plot_comparison", "plot_section_parameters",
]
