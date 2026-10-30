"""Automatic postprocessing of a SWAN project: tables of all the cases and map figures of chosen cases.

Set in the ``postprocess`` section of the project file (``postprocess_dir`` is the output
folder). Each call only does what is pending:

- ``points.xlsx``: every finished case × grid × output point (the output quantities and the sea
  state of the case), see :func:`~environmentaltools.models.swan.read_points`;
- ``convergence.xlsx``: one row per case and grid (accuracy reached, iterations, errors, minutes),
  see :func:`~environmentaltools.models.swan.summary`;
- ``maps/<case>.png``: the saved maps of the cases listed in ``postprocess: maps: cases``.

The two tables are written again when there are new results or the project file changed; a map
figure when its case was run again.
"""

from pathlib import Path

import pandas as pd

from ..common.tables import is_up_to_date, write_tables
from .read import _cases, read_points, summary


def postprocess(cfg, force=False, verbose=True):
    """Write the tables of all the finished cases and the map figures of the chosen cases.

    Args:
        cfg (SwanConfig): Configuration.
        force (bool): Write everything again.

    Returns:
        pd.DataFrame: What was done (``output``, ``status``).
    """
    out = cfg.postprocess_dir
    out.mkdir(parents=True, exist_ok=True)
    done = [cfg.results_dir / c / "done.json" for c in _cases(cfg)]
    sources = [cfg.file, cfg.variants_file, *done]
    rows = []
    if not done:
        if verbose:
            print("No finished cases yet")
        return pd.DataFrame(columns=["output", "status"])

    f = out / "points.xlsx"
    if (cfg["outputs"].get("points") is not None) and (force or not is_up_to_date(f, sources)):
        pts = read_points(cfg)
        write_tables(f, {"points": pts}, legend={
            "case, state, variant": "Case (sea state × variant)",
            "grid": "Grid of the result (the same point is given in every grid that contains it)",
            "point": "Output point (outputs: points of the project file)",
            "output columns": f"SWAN output quantities: {', '.join(cfg['outputs']['point_quantities'])} "
                              "(hs: significant wave height, tps: peak period, dir: mean direction, nautical…)",
            "state columns": "Sea state of the case at the boundary (Hs, Tp, Dir, Vv, DirV…)",
        })
        rows.append({"output": f.name, "status": f"written ({len(pts)} rows, {pts['case'].nunique()} cases)"})
    else:
        rows.append({"output": f.name, "status": "up to date" if f.exists() else "no output points"})

    f = out / "convergence.xlsx"
    if force or not is_up_to_date(f, sources):
        conv = summary(cfg)
        write_tables(f, {"convergence": conv}, legend={
            "accuracy": "Percentage of wet nodes that met the accuracy criterion (NUM ACCUR) in the last iteration",
            "iterations": "Iterations done",
            "minutes": "Run time of the grid",
            "errors": "Errors reported in PRINT",
        })
        rows.append({"output": f.name, "status": f"written ({len(conv)} rows)"})
    else:
        rows.append({"output": f.name, "status": "up to date"})

    maps = (cfg["postprocess"].get("maps") or {})
    finished = set(_cases(cfg))
    for case in maps.get("cases") or []:
        f = out / "maps" / f"{case}.png"
        if case not in finished:
            rows.append({"output": f"maps/{f.name}", "status": "no result"})
            continue
        if not force and is_up_to_date(f, [cfg.results_dir / case / "done.json", cfg.file]):
            rows.append({"output": f"maps/{f.name}", "status": "up to date"})
            continue
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("Agg")
            from .plot import plot_case_maps

            f.parent.mkdir(parents=True, exist_ok=True)
            plot_case_maps(cfg, case, quantity=maps.get("quantity"), file_name=f)
            plt.close("all")
            rows.append({"output": f"maps/{f.name}", "status": "written"})
        except FileNotFoundError as exc:      # maps kept only on the computing machine (store: archive)
            rows.append({"output": f"maps/{f.name}", "status": f"map not here: {exc}"[:120]})
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
        print(f"Results in {out}")
    return df
