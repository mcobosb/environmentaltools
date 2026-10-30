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

Workflow, from one project file (start from ``barrier_template.yaml`` in this folder):
a base case (a folder with BARRIER's ``config.yaml`` and its CSV files) and a table of
cases (``cases_file``, Excel: one row per case, one column per change of the base case)::

    from environmentaltools.models import barrier

    project = barrier.load_project("barrier.yaml")
    barrier.prepare(project)       # create/update the cases + overview to review them
    barrier.run(project)           # run the cases without result (resumable)
    barrier.postprocess(project)   # series at points, figures, comparison with a reference

or, without writing code::

    python -m environmentaltools.models.barrier all barrier.yaml

Each step does only what is pending: a case changed in the list (or a changed base
case) is created and run again, and only new results are postprocessed.

The pieces can also be used one by one (``create_case``, ``run_case``,
``read_output``, ``extract_points``, ``compare`` and the ``plot_*`` functions).
"""

from pathlib import Path

from .config import (BarrierProject, input_files, load_project, read_cases_table, read_config, start_date,
                     write_config)
from .inputs import (CROSS_SECTION_COLUMNS, MONOTONIC, check_cross_sections, fix_cross_sections, read_csv,
                     section_profile, set_section_parameter, write_csv, write_series, write_tributaries)
from .cases import (cases_overview, cases_to_table, create_case, folder_signature, prepare, write_cases_table,
                    write_overview)
from .run import barrier_executable, has_time_steps, is_readable, run, run_case
from .read import (COMPARE_TOLERANCES, compare, distance_to_mouth, extract_points, latest_output,
                   nearest_sections, read_output, read_project, tidal_mean, time_step_hours)
from .plot import plot_along_channel, plot_comparison, plot_point_series, plot_section_parameters
from .postprocess import load_points, postprocess, postprocess_output

TEMPLATE = Path(__file__).with_name("barrier_template.yaml")

__all__ = [
    "BarrierProject", "load_project", "read_cases_table", "cases_to_table", "write_cases_table",
    "read_config", "write_config", "input_files", "start_date", "TEMPLATE",
    "CROSS_SECTION_COLUMNS", "MONOTONIC", "read_csv", "write_csv", "write_series", "write_tributaries",
    "section_profile", "set_section_parameter", "check_cross_sections", "fix_cross_sections",
    "create_case", "prepare", "cases_overview", "write_overview", "folder_signature",
    "barrier_executable", "has_time_steps", "is_readable", "run_case", "run",
    "COMPARE_TOLERANCES", "read_output", "latest_output", "distance_to_mouth", "time_step_hours", "tidal_mean",
    "nearest_sections", "extract_points", "compare", "read_project",
    "plot_point_series", "plot_along_channel", "plot_comparison", "plot_section_parameters",
    "load_points", "postprocess", "postprocess_output",
]
