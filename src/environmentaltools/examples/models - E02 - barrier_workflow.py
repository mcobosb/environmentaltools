"""BARRIER workflow for any channel, in four blocks: the model → the cases → the runs → the analysis.

Start a project with ``python -m environmentaltools.models.barrier new <folder>``: it writes
``<folder>/barrier.yaml`` (every option explained, an idealised channel that runs as it comes) and
``<folder>/inputs/``. The project file is the only file to edit:

1. **The model**, the base case every case starts from: the sections ``geometry`` (a DEM and the
   channel axis, or an idealised channel) and ``simulation`` (period, tide, discharge, salinity) write
   it in ``base_case/`` (step ``base``). An existing base case (a folder with BARRIER's ``config.yaml``
   and its CSV files) can be used instead, without those sections: check its cross-section tables
   once (step 0 below).
2. **The cases**: ``cases.xlsx`` (``cases_file``), one row per case and one column per change of the
   base case, e.g. ``along_channel.nmann.scale`` or ``hydrodynamics.courant_number``; an empty cell
   means "as in the base case". The first ``prepare`` creates it with examples and a legend.
3. **The runs**: the BARRIER executable (https://github.com/mcobosb/Barrier), in the project file or
   in the environment variable ``BARRIER_EXE``.
4. **The analysis**: points, reference case and figures of ``postprocess``.

Then run the steps, from Python (below) or from the terminal::

    python -m environmentaltools.models.barrier base barrier.yaml
    python -m environmentaltools.models.barrier prepare barrier.yaml
    python -m environmentaltools.models.barrier run barrier.yaml
    python -m environmentaltools.models.barrier postprocess barrier.yaml
    python -m environmentaltools.models.barrier all barrier.yaml      # the four in a row

Every step only does what is pending, so they can be run again at any time: add or change a
row of ``cases.xlsx`` (or change the base case) and the same calls create it, run it and
postprocess it, leaving the rest as they are.
"""
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from environmentaltools.models import barrier  # noqa: E402

# 0. (once) the cross-section tables of the base case must be monotonic with the depth
cs = barrier.read_csv("base_case/cross-sections.csv")
if not barrier.check_cross_sections(cs).empty:
    cs, n = barrier.fix_cross_sections(cs)
    barrier.write_csv(cs, "base_case/cross-sections.csv")      # every digit is kept: rounding changes the results

# (optional) boundary series from dated data, written as BARRIER expects them
tide = pd.read_csv("tide.csv", index_col=0, parse_dates=True)["level"]
start = barrier.start_date(barrier.read_config("base_case"))
barrier.write_series(tide, "base_case/tides.csv", start=start, names=("time_seconds", "level_m"))

project = barrier.load_project("barrier.yaml")

# 1. BASE: the base case from the sections geometry and simulation (nothing to do without them)
barrier.prepare_base(project)

# 2. PREPARE: one folder per case + cases_overview.xlsx (base case and cases side by side, changes in
#    orange) and section_parameters.png in each case → review them before running
manifest = barrier.prepare(project)

# 3. RUN: cases without result (results_dir/<case>/: NetCDF, log, screen output, done.json)
log = barrier.run(project)

# 4. POSTPROCESS: new results only (postprocess_dir/<case>/: timeseries.xlsx and figures of the points,
#    the variable along the channel and, with a reference, the comparison with it)
summary = barrier.postprocess(project)
print(summary)

# the pieces are also available for anything else, e.g. one output and one point:
ds = barrier.read_output(project.results_dir / project.case_names[0])
points, series = barrier.extract_points(ds, {"Mouth": 0, "Gauge": 20})
print(series["S"].describe())
