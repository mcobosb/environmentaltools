"""SWAN project with a nested grid, driven by a YAML file.

1. Copy ``environmentaltools/models/swan/swan_template.yaml`` next to your data
   and edit grids, boundaries, outputs and sea states.
2. Prepare the cases (any computer), run them (computer with SWAN) and read the results.
"""
from environmentaltools.models import swan

cfg = swan.load_config("swan.yaml")

# figure of the grids, forced boundaries and output points
swan.plot_grids(cfg).figure.savefig("grids.png", dpi=150)

# one folder per sea state (× variant) with INPUT_<k>_<grid>; shared grid files in _common/
manifest = swan.prepare(cfg)

# run (resumable: finished cases have results/<case>/done.json); filter="..." runs a subset
swan.run(cfg)

# results
conv = swan.summary(cfg)                      # convergence, iterations, errors per case and grid
pts = swan.read_points(cfg)                   # long table: case, grid, point, hs, tps, dir, ... + state
hs = pts[pts.grid == cfg.grid_names[-1]].pivot(index="case", columns="point", values="hs")
print(conv.groupby("grid")[["accuracy", "iterations"]].describe())
print(hs.head())
