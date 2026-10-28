"""BARRIER estuary case: a variant of a base case, run, series at points and salt intrusion.

1. Start from a case folder with BARRIER's config.yaml and its CSV files (here ``run_001``).
2. Build the BARRIER executable (https://github.com/mcobosb/Barrier) and set BARRIER_EXE.
"""
import pandas as pd

from environmentaltools.models import barrier

# check the cross-section tables (and fix them if they are not monotonic with the depth)
cs = barrier.read_csv("run_001/cross-sections.csv")
if not barrier.check_cross_sections(cs).empty:
    cs, n = barrier.fix_cross_sections(cs)
    barrier.write_csv(cs, "run_001/cross-sections.csv")

# new case: another period with its tide, Kh from control points smoothed along the channel
tide = pd.read_csv("tide_2010.csv", index_col=0, parse_dates=True)["level"]
case = barrier.create_case(
    "run_002", "run_001",
    config={"run": {"start_date": "2010-01-01T00:00:00", "end_date": "2010-03-01T00:00:00"}},
    along_channel={"kh": {"points": [[0, 10], [50000, 100], [100000, 500]], "sigma": 50}})
barrier.write_series(tide, "run_002/tides.csv", start="2010-01-01", names=("time_seconds", "level_m"))
barrier.plot_section_parameters({"run_001": barrier.read_csv("run_001/along_channel_data.csv"),
                                 "run_002": barrier.read_csv("run_002/along_channel_data.csv")},
                                file_name="parameters.png")

# run and read
nc = barrier.run_case(case, "outputs/run_002")
ds = barrier.read_output(nc)
points, series = barrier.extract_points(ds, {"Mouth": 0, "Gauge A": 20, "Gauge B": 60})
series["eta"].to_csv("level_points.csv")
barrier.plot_point_series(series["S"], "salinity (psu)", "run_002", file_name="salinity_points.png")
barrier.plot_along_channel(ds, "S", points, title="Salt intrusion · run_002", file_name="salt_intrusion.png")

# comparison with the base case
ref = barrier.read_output("outputs/run_001")
summary, per_section = barrier.compare(ref, ds)
print(summary)
