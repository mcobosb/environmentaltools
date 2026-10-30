"""SWAN workflow: grids → project file + tables → prepare → run → postprocess.

A task is organised as::

    01_INPUTS/      what a person writes
        swan.yaml       project file: what hardly changes (grids and boundaries, outputs, folders, executable;
                        physics and numerics default to the validated set-up)
        states.xlsx     sea states: one row per state (Hs, Tp, Dir, wind…), e.g. the cases selected with MDA
        variants.xlsx   (optional) variants: one row per variant, one column per changed key written with dots
                        (physics.wcap = WCAP KOMEN delta=1, boundary_spectrum.spreading_value = 2…); an empty
                        cell keeps the base value and 'off' removes a command
        points.csv      output points (name, x, y)
    02_SCRIPTS/     one short script per step (00 grids, 01 prepare, 02 run, 03 postprocess)
    03_OUTPUTS/     what the scripts write (cases, results, overview, postprocess)

Start ``swan.yaml`` from ``environmentaltools/models/swan/swan_template.yaml``.

The cases are every sea state × every active variant (``active_variants`` in ``swan.yaml``, all by default),
one folder per variant: ``<variant>/<state>`` (without variants, just ``<state>``). Every step only does what
is pending, so they can be run again at any time: change a row of a table (or the grids) and ``prepare``
writes those cases again, ``run`` runs them and ``postprocess`` updates the tables; switching a variant on
or off leaves the cases of the others as they are.

The same steps from the terminal::

    python -m environmentaltools.models.swan prepare swan.yaml
    python -m environmentaltools.models.swan run swan.yaml                    # where SWAN is installed
    python -m environmentaltools.models.swan run swan.yaml --variant komen1   # only the cases of a variant
    python -m environmentaltools.models.swan run swan.yaml --case base/caso_001
    python -m environmentaltools.models.swan postprocess swan.yaml
    python -m environmentaltools.models.swan all swan.yaml                    # the three in a row
"""
import matplotlib

matplotlib.use("Agg")
from environmentaltools.models import swan  # noqa: E402

PROJECT = "01_INPUTS/swan.yaml"

# 0. GRIDS (once, before writing the boundaries). With only the grids in swan.yaml:
#      grids:
#        ext: {grd: grids/ext.grd, dep: grids/ext.dep}
#        fina: {grd: grids/fina.grd, dep: grids/fina.dep, nested_in: ext}
#    draw them and list the stretches of water along their sides (side, segment_ij, depths, length). The open sea
#    is usually the longest, deepest stretch: copy its segment_ij to   ext: boundaries: - segment_ij: [i1, j1, i2, j2]
stretches = swan.inspect_grids(PROJECT, points="01_INPUTS/points.csv", file_name="03_OUTPUTS/00_grids.png",
                               table_file="03_OUTPUTS/00_boundary_stretches.xlsx")
print(stretches)

cfg = swan.load_config(PROJECT)
swan.plot_grids(cfg).figure.savefig("03_OUTPUTS/00_grids_and_boundaries.png", dpi=150)   # check the boundaries

# 1. PREPARE (any computer): one folder per case, <variant>/<state>/ (INPUT_<k>_<grid> + case.json; grid files
#    in _common/), and cases_overview.xlsx: every case with its status (created / updated / kept) and whether it
#    has a result, and the active variants side by side with what each one changes in orange → review it
manifest = swan.prepare(cfg)
print(manifest["status"].value_counts())

# 2. RUN (computer with SWAN): the cases without result of the active variants; each one runs on local disk
#    (run_dir) and its outputs are copied to results_dir/<variant>/<state>/ with done.json. Interrupted? Run it
#    again: it goes on where it stopped
swan.run(cfg)                                  # or swan.run(cfg, variants=["komen1"]), cases=["base/caso_001"]

# 3. POSTPROCESS: postprocess_dir/points.xlsx (every case × grid × point, with its variant and sea state),
#    convergence.xlsx and maps/<variant>/<state>.png of the cases in postprocess: maps: cases
swan.postprocess(cfg)

# the results can also be read directly, e.g. Hs at every point of the last grid, one column per variant
pts = swan.read_points(cfg)
last = pts[pts.grid == cfg.grid_names[-1]]
print(last.pivot_table(index=["state", "point"], columns="variant", values="hs").head())
