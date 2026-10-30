"""Generation of SWAN cases (sea states × variants) ready to run."""

import json
import shutil
from pathlib import Path

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
        states = (pd.read_excel(f, sheet_name=st.get("sheet") or 0) if f.suffix.lower() in (".xlsx", ".xls")
                  else pd.read_csv(f))
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
    """List of ``(case, state_id, variant)``: ``<variant>/<state>`` (one folder per active variant) or, without
    variants, ``<state>``."""
    variants = cfg.variant_names or [None]
    return [(sid if v is None else f"{v}/{sid}", sid, v) for v in variants for sid in states.index]


def case_folders(root, marker="case.json"):
    """Case names (``<state>`` or ``<variant>/<state>``) of the folders of ``root`` that contain ``marker``."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != COMMON):
        if (d / marker).exists():
            out.append(d.name)
        else:
            out += [f"{d.name}/{s.name}" for s in sorted(d.iterdir()) if s.is_dir() and (s / marker).exists()]
    return out


def _same(path, text):
    return path.exists() and path.read_text() == text


def prepare(cfg, states=None, overwrite=False, overview=True, verbose=True):
    """Write all SWAN cases of a project; only what changed is written again.

    Layout of ``cases_dir``::

        _common/          grid coordinates and bottom files, nesting and output points
        <case>/           INPUT_<k>_<grid> (one per grid, in computation order) + case.json
        cases.csv         manifest (case, state, variant, status and state values)

    Every case is built in memory and compared with the one on disk:

    - ``created``: new case (a new row of the states table or a new variant);
    - ``updated``: its INPUT files differ (its state or variant changed), or the grid files of
      ``_common`` changed (then every case is updated). Its result is marked as pending (its
      ``done.json`` is removed), so :func:`run` runs it again;
    - ``kept``: identical, nothing is written (with ``overwrite`` it is written again, but its
      result is kept, since the content is the same).

    Case folders no longer in the list are reported, not removed. With ``overview``, the cases,
    their status and the variants side by side are written to ``cfg.overview_file`` to review
    them before running.

    Args:
        cfg (SwanConfig): Configuration.
        states (pd.DataFrame | None): Sea states (otherwise read from the YAML).
        overwrite (bool): Write every case again, changed or not.
        overview (bool): Write the overview Excel file.

    Returns:
        pd.DataFrame: Manifest of the cases (with a ``status`` column).
    """
    import tempfile

    states = load_states(cfg, states)
    root = cfg.cases_dir
    common = root / COMMON
    common.mkdir(parents=True, exist_ok=True)
    grids = cfg.grids()
    depmin = cfg["set"]["depmin"]

    with tempfile.TemporaryDirectory() as tmp:          # common files built aside, copied only if they changed
        tmp = Path(tmp)
        files = {}
        for name in cfg.grid_names:
            files[name] = write_grid_files(grids[name], tmp, depmin)
            files[name]["nest"] = {}
        for name in cfg.grid_names:
            parent = cfg["grids"][name].get("nested_in")
            if parent:
                nest = write_nest_points(grids[name], tmp)
                files[parent]["nest"][name] = nest
                files[name]["nest_in"] = f"nest_{name}.sp2"
        points = None
        if cfg["outputs"].get("points") is not None:
            pt = cfg["outputs"]["points"]
            points = pd.read_csv(cfg.path(pt)) if isinstance(pt, str) else pd.DataFrame(pt)
            if "name" not in points:
                points["name"] = [f"P{k + 1}" for k in range(len(points))]
            write_points(points, tmp)
            points[["name", "x", "y"]].to_csv(tmp / "points.csv", index=False)
            for name, g in grids.items():
                for r in points.itertuples():
                    if not g.contains(r.x, r.y) and verbose:
                        print(f"  note: point {r.name} is outside grid '{name}' (no values there)")
        changed_common = [f.name for f in tmp.iterdir()
                          if not ((common / f.name).exists() and (common / f.name).read_bytes() == f.read_bytes())]
        for f in tmp.iterdir():
            if f.name in changed_common or overwrite:
                shutil.copy2(f, common / f.name)
    existing_cases = bool(case_folders(root))
    if changed_common and existing_cases and verbose:
        print(f"Grid/point files changed ({', '.join(sorted(changed_common))}): every case is updated")

    rows = []
    position = {sid: n for n, sid in enumerate(states.index, 1)}     # run number of SWAN: the row of the state
    for case, sid, var in case_names(cfg, states):
        k = position[sid]
        c = cfg.variant(var)
        d = root / case
        st = {k_: (None if pd.isna(v) else float(v)) for k_, v in states.loc[sid].items()
              if not str(k_).startswith("in_")}
        st["case"] = case
        stages, texts = [], {}
        for n, name in enumerate(c.grid_names, 1):
            fname = f"INPUT_{n}_{name}"
            texts[fname] = build_input(c, grids[name], st, files[name], c.children(name), points is not None, str(k))
            stages.append({"grid": name, "input": fname})
        meta = {"case": case, "state": sid, "variant": var, "stages": stages, "values": st,
                "outputs": {g: {"points": f"points_{g}.tab" if points is not None else None,
                                "map": f"map_{g}.tab" if (c["outputs"].get("maps") or {}).get(g) else None,
                                "spectra": f"spec_{g}.sp2" if points is not None and c["outputs"].get(
                                    "point_spectra") else None} for g in c.grid_names}}
        meta_text = json.dumps(meta, indent=1)
        if not d.exists():
            status = "created"
        elif changed_common or not _same(d / "case.json", meta_text) or not all(
                (d / f).exists() and _same_input(d / f, t) for f, t in texts.items()):
            status = "updated"
        else:
            status = "kept"
        if status != "kept" or overwrite:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            for fname, text in texts.items():
                write_input(d / fname, text)
            (d / "case.json").write_text(meta_text)
        if status == "updated":
            (cfg.results_dir / case / "done.json").unlink(missing_ok=True)
        rows.append({"case": case, "state": sid, "variant": var, "status": status})
    man = pd.DataFrame(rows).merge(states, left_on="state", right_index=True, how="left")
    man.to_csv(root / "cases.csv", index=False)
    listed = set(man["case"])
    others = [c for c in case_folders(root) if c not in listed]
    if verbose:
        counts = man["status"].value_counts().to_dict()
        print(f"{len(man)} cases in {root} ({len(states)} states × {len(cfg.variant_names) or 1} variants; "
              f"grids: {' → '.join(cfg.grid_names)}): "
              + ", ".join(f"{counts.get(s, 0)} {s}" for s in ("created", "updated", "kept")))
        if counts.get("updated"):
            print("  the updated cases are pending again (run them with run)")
        if others:
            print(f"  folders no longer in the list (not removed): {len(others)} ({', '.join(others[:5])}…)")
    if overview:
        write_overview(cfg, man)
        if verbose:
            print(f"Review before running: {cfg.overview_file}")
    return man


def _same_input(path, text):
    """True if the command file ``path`` has the content ``text`` (written with CRLF by :func:`write_input`)."""
    return path.read_bytes().replace(b"\r\n", b"\n") == text.replace("\r\n", "\n").encode("latin-1")


def write_overview(cfg, manifest=None, file_name=None):
    """Excel file to review the cases before running.

    Sheets: ``cases`` (one row per case: state, variant, status of the last :func:`prepare`,
    whether it has a result, and the sea state), ``variants`` (every value of the configuration
    that a variant can change, base and variants side by side, changes in orange) and ``grids``.

    Returns:
        Path: The written file (``cfg.overview_file`` by default).
    """
    from ..common.tables import flatten, write_tables

    man = manifest if manifest is not None else pd.read_csv(cfg.cases_dir / "cases.csv")
    cases = man.copy()
    cases.insert(4, "result", ["done" if (cfg.results_dir / c / "done.json").exists() else "pending"
                               for c in cases["case"]])
    sections = ["spectral_grid", "set", "boundary_spectrum", "physics", "numerics", "outputs"]
    flat = {"base": flatten({k: cfg[k] for k in sections})}
    for v in cfg.variant_names:
        flat[v] = flatten({k: cfg.variant(v)[k] for k in sections})
    keys = list(dict.fromkeys(k for f in flat.values() for k in f))
    variants = pd.DataFrame({"parameter": keys, **{n: [str(f.get(k, "")) for k in keys] for n, f in flat.items()}})
    grids = pd.DataFrame([{"grid": g, "nested_in": cfg["grids"][g].get("nested_in") or "",
                           "nodes": "x".join(map(str, cfg.grids()[g].shape)),
                           "source": cfg["grids"][g].get("grd") or "regular"} for g in cfg.grid_names])
    names = cfg.variant_names
    return write_tables(file_name or cfg.overview_file, {"cases": cases, "variants": variants, "grids": grids},
                        highlight={"variants": ("base", names[0])} if names else None, legend={
        "sheet cases": "One row per case (sea state × active variant); its folder is <variant>/<state>",
        "status": "What the last prepare did: created (new), updated (its state, variant or the grids changed: its "
                  "result is pending again) or kept (unchanged)",
        "result": "done: finished (done.json in results_dir); pending: run will run it",
        "sheet variants": "Every value of the configuration a variant can change: base and active variants side by "
                          "side. In orange, what a variant changes",
        "sheet grids": "Grids in computation order",
        "states": f"Sea states: {cfg['states'].get('file')}",
        "variants table": f"{cfg.variants_file}" if cfg.variants_file else "variants of the project file",
    })


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
