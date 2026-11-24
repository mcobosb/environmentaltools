"""Minimal BARRIER project: a 20 km rectangular channel forced by an M2 tide at the mouth.

Writes a base case (geometry with prismatic_geometry, config.yaml and boundary series with
new_base_case) and a project file with two cases (the base case and Manning n x 1.2), then runs
prepare -> run -> postprocess.
Takes a few seconds. The BARRIER executable is taken from the environment variable
BARRIER_EXE (or 'barrier' on the PATH)::

    export BARRIER_EXE=/path/to/Barrier/build/barrier
    python minimal_example.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import yaml  # noqa: E402

from environmentaltools.models import barrier  # noqa: E402

base = Path("base_case")

# geometry: rectangular channel 300 m wide and 20 km long, bed at -5 m, tables up to +5 m every 0.25 m
barrier.prismatic_geometry(20000, dx=500, bottom_width=300, bed=-5, top_level=5, level_step=0.25, nmann=0.025,
                           kh=100, out_dir=base)

# config.yaml with safe settings and the boundary series: M2 tide of 1 m at the mouth, 50 m3/s from upstream,
# sea water of 36 psu
barrier.new_base_case(base, "2020-01-01", "2020-01-05", tide={"amplitude": 1.0}, discharge=50, salinity=36.0)

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
