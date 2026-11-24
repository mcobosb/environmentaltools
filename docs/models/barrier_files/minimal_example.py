"""Minimal BARRIER project: a 20 km rectangular channel forced by an M2 tide at the mouth.

Writes a base case (geometry, boundary series and config.yaml) and a project file with two
cases (the base case and Manning n x 1.2), then runs prepare -> run -> postprocess.
Takes a few seconds. The BARRIER executable is taken from the environment variable
BARRIER_EXE (or 'barrier' on the PATH)::

    export BARRIER_EXE=/path/to/Barrier/build/barrier
    python minimal_example.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from environmentaltools.models import barrier  # noqa: E402

base = Path("base_case")
base.mkdir(exist_ok=True)

# along-channel data: one row per section, x from the upstream boundary (0) to the mouth (20 km).
# BARRIER reads the columns by position: x, z, nmann, xutm, yutm, AngMd, AngMi, beta, Sf, kh
x = np.arange(0, 20001, 500.0)
along = pd.DataFrame({"x": x, "z": -5.0, "nmann": 0.025, "xutm": x, "yutm": 0.0,
                      "AngMd": 0.0, "AngMi": 0.0, "beta": 1.0, "Sf": 1.0, "kh": 100.0})
barrier.write_csv(along, base / "along_channel_data.csv")

# cross sections: rectangular, 300 m wide, tabulated every 0.25 m of water depth
width = 300.0
rows = []
for xi in x:
    for depth in np.arange(0, 10.01, 0.25):
        area, perimeter = width * depth, width + 2 * depth
        rows.append((xi, -5.0, depth, width, area, perimeter, area / perimeter, width, -width / 2, width / 2, 1.0))
cs = pd.DataFrame(rows, columns=barrier.CROSS_SECTION_COLUMNS)
assert barrier.check_cross_sections(cs).empty          # monotonic with the depth
barrier.write_csv(cs, base / "cross-sections.csv")

config = {
    "run": {"name": "minimal", "start_date": "2020-01-01T00:00:00", "end_date": "2020-01-05T00:00:00",
            "timestep": 1800, "output_variables": ["eta", "Q", "A", "U", "S"], "log_level": 0},
    "geometry": {"along_channel_file": "along_channel_data.csv", "cross_sections_file": "cross-sections.csv"},
    "hydrodynamics": {
        "initial_conditions": {"type": "calm"},
        "boundary_conditions": {"upstream": {"type": "discharge", "file": "discharge.csv"},
                                "downstream": {"type": "elevation", "file": "tides.csv"}},
        "forcing": {"tributaries_file": None},
        "courant_number": 0.5,
        "tvd_limiter": {"enabled": True, "method": "minmod", "psi_formula": "tseng", "delta": 0.3},
        "dry_bed": True},
    "transport": {"salinity": {"enabled": True, "dispersion_kh": 100.0,
                               "boundary_conditions": {"downstream": {"type": "ocean", "value": 36.0}}}},
}
(base / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

# boundary series indexed by date, written as seconds since run.start_date
start = barrier.start_date(config)
dates = pd.date_range(start, "2020-01-05", freq="10min")
hours = (dates - start).total_seconds() / 3600
tide = pd.Series(1.0 * np.sin(2 * np.pi * hours / 12.42), index=dates)       # M2, 1 m amplitude
barrier.write_series(tide, base / "tides.csv", start=start, names=("time_seconds", "eta_m"))
barrier.write_series(pd.Series(50.0, index=dates), base / "discharge.csv", start=start,
                     names=("time_seconds", "Q_m3s"))

project_file = Path("barrier.yaml")
project_file.write_text(yaml.safe_dump({
    "base_case": "base_case",
    "cases": {"reference": {},
              "rough": {"along_channel": {"nmann": {"scale": 1.2}}}},
    "postprocess": {"points": {"Mouth": 0, "Middle": 10, "Head": 19},
                    "reference": {"case": "reference"}},
}, sort_keys=False))

project = barrier.load_project(project_file)
barrier.prepare(project)
barrier.run(project)
print(barrier.postprocess(project))
