"""SWAN workflow: base configuration + sea states table (+ variants table) → prepare → run → postprocess.

What a project needs:

1. A **project file** ``swan.yaml``: copy ``environmentaltools/models/swan/swan_template.yaml`` and set
   the grids (Delft3D ``.grd``/``.dep`` or regular + bathymetry; a nested grid gives ``nested_in``),
   the boundaries, the physics and numerics (validated defaults), the outputs (points, maps), the
   folders and the executable (or a list, one per computer; or ``auto``). This is what hardly changes.
2. A **sea states table** (``states: file``, csv or Excel): one row per state with Hs, Tp, Dir, wind…
   (column names mapped in ``states: columns``), e.g. the cases selected with MDA.
3. Optionally, a **variants table** (``variants_file``, Excel): one row per variant and one column per
   changed key, e.g. ``physics.wcap`` = ``WCAP KOMEN delta=1`` or ``boundary_spectrum.spreading_value``
   = 2; an empty cell keeps the base value and ``off`` removes a command. The cases are states × variants.

Then run the three steps, from Python (below) or from the terminal::

    python -m environmentaltools.models.swan prepare swan.yaml
    python -m environmentaltools.models.swan run swan.yaml            # where SWAN is installed
    python -m environmentaltools.models.swan postprocess swan.yaml
    python -m environmentaltools.models.swan all swan.yaml            # the three in a row

Every step only does what is pending, so they can be run again at any time: change a row of a table
(or the grids) and prepare writes those cases again, run runs them and postprocess updates the tables.
"""
import matplotlib

matplotlib.use("Agg")
from environmentaltools.models import swan  # noqa: E402

cfg = swan.load_config("swan.yaml")

# grids, forced boundaries and output points (check them once)
swan.plot_grids(cfg).figure.savefig("grids.png", dpi=150)

# 1. PREPARE: one folder per case (INPUT_<k>_<grid> + case.json; grid files in _common/) and
#    cases_overview.xlsx: every case with its status (created / updated / kept) and the variants side by side
manifest = swan.prepare(cfg)
print(manifest["status"].value_counts())

# 2. RUN (computer with SWAN): cases without result; each one runs on local disk (run_dir) and its outputs
#    are copied to results_dir/<case> with done.json. Interrupted? Run it again: it goes on where it stopped
swan.run(cfg)                      # swan.run(cfg, cases=["caso_001"]) or filter="caso_1" for a subset

# 3. POSTPROCESS: postprocess_dir/points.xlsx (every case × grid × point, with its sea state),
#    convergence.xlsx and maps/<case>.png of the cases in postprocess: maps: cases
swan.postprocess(cfg)

# the results can also be read directly, e.g. Hs at every point of the last grid
pts = swan.read_points(cfg)
hs = pts[pts.grid == cfg.grid_names[-1]].pivot(index="case", columns="point", values="hs")
print(hs.head())
