"""BARRIER workflow for any channel: base case → list of cases → prepare → run → postprocess.

What a project needs (one folder, e.g. ``01_INPUTS/``):

1. A **base case**: a folder with BARRIER's ``config.yaml`` and the CSV files it names
   (``along_channel_data.csv``, ``cross-sections.csv``, tide, discharge…). Check the
   cross-section tables once (step 0 below).
2. A **project file** ``barrier.yaml``: copy
   ``environmentaltools/models/barrier/barrier_template.yaml`` and set the base case, the
   folders, the executable (or a list, one per computer), the list of cases (the base
   case plus changes) and the postprocessing (points, reference, figures).
3. The BARRIER executable (https://github.com/mcobosb/Barrier), in the project file or in
   the environment variable ``BARRIER_EXE``.

Then run the three steps, from Python (below) or from the terminal::

    python -m environmentaltools.models.barrier prepare barrier.yaml
    python -m environmentaltools.models.barrier run barrier.yaml
    python -m environmentaltools.models.barrier postprocess barrier.yaml
    python -m environmentaltools.models.barrier all barrier.yaml      # the three in a row

Every step only does what is pending, so they can be run again at any time: add or change a
case in ``barrier.yaml`` (or change the base case) and the same three calls create it, run it
and postprocess it, leaving the rest as they are.
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

# 1. PREPARE: one folder per case + cases_overview.xlsx (base case and cases side by side, changes in
#    orange) and section_parameters.png in each case → review them before running
manifest = barrier.prepare(project)

# 2. RUN: cases without result (results_dir/<case>/: NetCDF, log, screen output, done.json)
log = barrier.run(project)

# 3. POSTPROCESS: new results only (postprocess_dir/<case>/: timeseries.xlsx and figures of the points,
#    the variable along the channel and, with a reference, the comparison with it)
summary = barrier.postprocess(project)
print(summary)

# the pieces are also available for anything else, e.g. one output and one point:
ds = barrier.read_output(project.results_dir / project.case_names[0])
points, series = barrier.extract_points(ds, {"Mouth": 0, "Gauge": 20})
print(series["S"].describe())
