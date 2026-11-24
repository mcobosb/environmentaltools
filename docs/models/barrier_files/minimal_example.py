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

# geometry: rectangular channel 300 m wide and 20 km long, bed at -5 m, tables up to +5 m every 0.25 m
barrier.prismatic_geometry(20000, dx=500, bottom_width=300, bed=-5, top_level=5, level_step=0.25, nmann=0.025,
                           kh=100, out_dir=base)

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
