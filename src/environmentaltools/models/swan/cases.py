"""Generation of SWAN cases (sea states × variants) ready to run."""

import json
import shutil

import numpy as np
import pandas as pd

from .inputs import build_input, write_grid_files, write_input, write_nest_points, write_points

COMMON = "_common"


def load_states(cfg, states=None):
    """Sea states as a DataFrame with standard columns ``Hs, Tp, Dir, Vv, DirV[, spreading, level]``.

    Args:
        cfg (SwanConfig): Configuration (``states`` section gives the file, the
            id column and the column names).
        states (pd.DataFrame | None): States given directly (same column mapping).

    Returns:
        pd.DataFrame: Indexed by state id (str); the original columns are kept
        with the prefix ``in_``.
    """
    st = cfg["states"]
    if states is None:
        f = cfg.path(st["file"])
        if f is None:
            raise ValueError("No states: give 'states: file:' in the YAML or a DataFrame")
        states = pd.read_excel(f) if f.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(f)
    df = states.copy()
    if st.get("id"):
        df = df.set_index(st["id"])
    df.index = df.index.map(str)
    out = pd.DataFrame(index=df.index)
    for std, col in st["columns"].items():
        if col is not None and col in df:
            out[std] = df[col].astype(float)
    missing = {"Hs", "Tp", "Dir"} - set(out)
    if missing:
        raise ValueError(f"States without columns {missing} (check states: columns:)")
    return out.join(df.add_prefix("in_"))


def case_names(cfg, states):
    """List of ``(case, state_id, variant)``."""
    variants = list(cfg["variants"]) or [None]
    return [(sid if v is None else f"{sid}_{v}", sid, v) for sid in states.index for v in variants]


def prepare(cfg, states=None, overwrite=False, verbose=True):
    """Write all SWAN cases of a project.

    Layout of ``cases_dir``::

        _common/          grid coordinates and bottom files, nesting and output points
        <case>/           INPUT_<k>_<grid> (one per grid, in computation order) + case.json
        cases.csv         manifest (case, state, variant and state values)

    Args:
        cfg (SwanConfig): Configuration.
        states (pd.DataFrame | None): Sea states (otherwise read from the YAML).
        overwrite (bool): Rewrite existing case folders.

    Returns:
        pd.DataFrame: Manifest of the cases.
    """
    states = load_states(cfg, states)
    root = cfg.cases_dir
    common = root / COMMON
    common.mkdir(parents=True, exist_ok=True)
    grids = cfg.grids()
    depmin = cfg["set"]["depmin"]

    files = {}
    for name in cfg.grid_names:
        files[name] = write_grid_files(grids[name], common, depmin)
        files[name]["nest"] = {}
    for name in cfg.grid_names:
        parent = cfg["grids"][name].get("nested_in")
        if parent:
            nest = write_nest_points(grids[name], common)
            files[parent]["nest"][name] = nest
            files[name]["nest_in"] = f"nest_{name}.sp2"
    points = None
    if cfg["outputs"].get("points") is not None:
        p = cfg["outputs"]["points"]
        points = pd.read_csv(cfg.path(p)) if isinstance(p, str) else pd.DataFrame(p)
        if "name" not in points:
            points["name"] = [f"P{k + 1}" for k in range(len(points))]
        write_points(points, common)
        points[["name", "x", "y"]].to_csv(common / "points.csv", index=False)
        for name, g in grids.items():
            for r in points.itertuples():
                if not g.contains(r.x, r.y) and verbose:
                    print(f"  note: point {r.name} is outside grid '{name}' (no values there)")

    rows = []
    for k, (case, sid, var) in enumerate(case_names(cfg, states), 1):
        c = cfg.variant(var)
        d = root / case
        if d.exists() and not overwrite:
            rows.append({"case": case, "state": sid, "variant": var})
            continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        st = {k_: (None if pd.isna(v) else float(v)) for k_, v in states.loc[sid].items()
              if not str(k_).startswith("in_")}
        st["case"] = case
        stages = []
        for n, name in enumerate(c.grid_names, 1):
            text = build_input(c, grids[name], st, files[name], c.children(name), points is not None, str(k))
            fname = f"INPUT_{n}_{name}"
            write_input(d / fname, text)
            stages.append({"grid": name, "input": fname})
        meta = {"case": case, "state": sid, "variant": var, "stages": stages, "values": st,
                "outputs": {g: {"points": f"points_{g}.tab" if points is not None else None,
                                "map": f"map_{g}.tab" if (c["outputs"].get("maps") or {}).get(g) else None,
                                "spectra": f"spec_{g}.sp2" if points is not None and c["outputs"].get(
                                    "point_spectra") else None} for g in c.grid_names}}
        (d / "case.json").write_text(json.dumps(meta, indent=1))
        rows.append({"case": case, "state": sid, "variant": var})
    man = pd.DataFrame(rows).merge(states, left_on="state", right_index=True, how="left")
    man.to_csv(root / "cases.csv", index=False)
    if verbose:
        print(f"{len(man)} cases in {root} ({len(states)} states × {len(cfg['variants']) or 1} variants; "
              f"grids: {' → '.join(cfg.grid_names)})")
    return man


def boundary_distances(cfg, grid_name, boundary=0):
    """Distance (m) along a ``segment_ij`` boundary: helper to build ``variable:`` points.

    Returns:
        pd.DataFrame: ``j, i, x, y, depth, s`` of the boundary nodes.
    """
    g = cfg.grids()[grid_name]
    i1, j1, i2, j2 = cfg["grids"][grid_name]["boundaries"][boundary]["segment_ij"]
    n = max(abs(i2 - i1), abs(j2 - j1)) + 1
    i = np.linspace(i1, i2, n).round().astype(int)
    j = np.linspace(j1, j2, n).round().astype(int)
    return pd.DataFrame({"j": j, "i": i, "x": g.x[j, i], "y": g.y[j, i],
                         "depth": None if g.depth is None else g.depth[j, i], "s": g.distance_along(j, i)})
