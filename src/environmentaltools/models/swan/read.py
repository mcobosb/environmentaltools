"""Readers of SWAN results: tables, 2-D spectra and PRINT diagnostics, per case or for a whole project."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

#: Output columns of each SWAN quantity keyword (vectors give two columns).
QUANTITY_COLUMNS = {
    "XP": ["x"], "YP": ["y"], "DEPTH": ["depth"], "HSIGN": ["hs"], "HSWELL": ["hswell"], "TPS": ["tps"],
    "RTP": ["rtp"], "TM01": ["tm01"], "TM02": ["tm02"], "TMM10": ["tmm10"], "PER": ["per"], "RPER": ["rper"],
    "DIR": ["dir"], "PDIR": ["pdir"], "TDIR": ["tdir"], "DSPR": ["dspr"], "WLENGTH": ["wlength"],
    "STEEPW": ["steepw"], "QB": ["qb"], "SETUP": ["setup"], "UBOT": ["ubot"], "URMS": ["urms"],
    "DISSIP": ["dissip"], "DISWCAP": ["diswcap"], "DISBOT": ["disbot"], "DISSURF": ["dissurf"],
    "DISVEG": ["disveg"], "LEAK": ["leak"], "DHSIGN": ["dhsign"], "DRTM01": ["drtm01"], "DIST": ["dist"],
    "WIND": ["wind_x", "wind_y"], "VELOC": ["vel_x", "vel_y"], "TRANSP": ["transp_x", "transp_y"],
    "FORCE": ["force_x", "force_y"], "FORCES": ["force_x", "force_y"],
}
EXCEPTION_VALUES = (-9.0, -99.0, -999.0, -9999.0)


def quantity_columns(quantities):
    cols = []
    for q in quantities:
        cols += QUANTITY_COLUMNS.get(q.upper(), [q.lower()])
    return cols


def read_table(file_name, quantities=None):
    """Read a SWAN ``TABLE`` file (``HEAD`` or ``NOHEAD``).

    Args:
        file_name (str | Path): Table file.
        quantities (list[str] | None): SWAN quantity keywords in output order
            (gives standard column names: ``hs``, ``tps``, ``dir``, ``wind_x``…).
            Without it, the header names of a ``HEAD`` table are used.

    Returns:
        pd.DataFrame: One row per location; exception values as ``NaN``.
    """
    lines = Path(file_name).read_text(errors="ignore").splitlines()
    head = [l for l in lines if l.startswith("%")]
    data = np.loadtxt([l for l in lines if l.strip() and not l.startswith("%")], ndmin=2)
    if quantities:
        cols = quantity_columns(quantities)
    else:
        names = [l[1:].split() for l in head if l[1:].split() and not l[1:].strip().startswith(("[", "Run"))]
        cols = names[0] if names else [f"c{k}" for k in range(data.shape[1])]
    if len(cols) != data.shape[1]:
        raise ValueError(f"{file_name}: {data.shape[1]} columns but {len(cols)} names ({cols})")
    df = pd.DataFrame(data, columns=cols)
    return df.mask(df.isin(EXCEPTION_VALUES) & ~df.columns.isin(["x", "y", "Xp", "Yp"]))


def read_print(file_name):
    """Diagnostics of a SWAN ``PRINT`` file.

    Returns:
        dict: ``version``, ``iterations``, ``accuracy`` (last % of wet points
        meeting the criterion), ``required`` (%), ``converged``, ``errors``,
        ``warnings`` and ``messages`` (first lines of errors/warnings).
    """
    txt = Path(file_name).read_text(errors="ignore")
    acc = re.findall(r"accuracy OK in\s+([\d.]+)\s*% of wet grid points \(\s*([\d.]+)\s*% required", txt)
    it = re.findall(r"iteration\s+(\d+);", txt)
    ver = re.search(r"VERSION NUMBER\s+([\w.]+)|SWAN version[:\s]+([\w.]+)", txt)
    msgs = [l.strip() for l in txt.splitlines() if re.search(r"\*\* (Error|Warning|Severe)", l)]
    last = float(acc[-1][0]) if acc else np.nan
    req = float(acc[-1][1]) if acc else np.nan
    return {"version": (ver.group(1) or ver.group(2)) if ver else None,
            "iterations": int(it[-1]) if it else 0, "accuracy": last, "required": req,
            "converged": bool(acc) and last >= req,
            "errors": sum("Error" in m or "Severe" in m for m in msgs),
            "warnings": sum("Warning" in m for m in msgs), "messages": msgs[:10]}


def read_spectra(file_name, rho=1025.0, grav=9.81):
    """Read a SWAN 2-D spectrum file (``SPEC … SPEC2D``, stationary).

    Energy densities (``EnDens``, J/m²/Hz/deg, written when ``INRHOG = 1``) are
    converted to variance densities dividing by ``rho · grav``.

    Returns:
        dict: ``x``, ``y`` (locations), ``freq`` (Hz), ``dir`` (deg),
        ``direction_convention`` and ``E`` with shape ``(n_locations, n_freq,
        n_dir)`` (variance density m²/Hz/deg; ``NaN`` for NODATA).
    """
    tok = Path(file_name).read_text(errors="ignore").splitlines()
    k = 0

    def nxt():
        nonlocal k
        while not tok[k].strip() or tok[k].lstrip().startswith("$"):
            k += 1
        line = tok[k]
        k += 1
        return line

    out = {}
    while k < len(tok):
        line = nxt().split()
        key = line[0].upper()
        if key in ("LOCATIONS", "LONLAT"):
            n = int(nxt().split()[0])
            xy = np.array([nxt().split()[:2] for _ in range(n)], float)
            out["x"], out["y"] = xy[:, 0], xy[:, 1]
        elif key in ("AFREQ", "RFREQ"):
            n = int(nxt().split()[0])
            out["freq"] = np.array([float(nxt().split()[0]) for _ in range(n)])
        elif key in ("NDIR", "CDIR"):
            n = int(nxt().split()[0])
            out["dir"] = np.array([float(nxt().split()[0]) for _ in range(n)])
            out["direction_convention"] = "nautical" if key == "NDIR" else "cartesian"
        elif key == "QUANT":
            for _ in range(int(nxt().split()[0])):
                quant = nxt().split()[0]
                nxt()
                exc = float(nxt().split()[0])
            nl, nf, nd = len(out["x"]), len(out["freq"]), len(out["dir"])
            E = np.full((nl, nf, nd), np.nan)
            loc = 0
            while loc < nl and k < len(tok):
                w = nxt().split()
                if w[0].upper() == "FACTOR":
                    fac = float(nxt().split()[0])
                    E[loc] = np.array([nxt().split() for _ in range(nf)], float) * fac
                    E[loc][E[loc] == exc * fac] = np.nan
                elif w[0].upper() == "ZERO":
                    E[loc] = 0.0
                elif not w[0].upper() == "NODATA":
                    continue
                loc += 1
            out["E"] = E / (rho * grav) if quant.lower().startswith("endens") else E
            break
    return out


def spectrum_parameters(spec):
    """Hs, peak period and mean direction (nautical) of each location of :func:`read_spectra`.

    Returns:
        pd.DataFrame: ``x, y, hs, tp, dir``.
    """
    E, f, d = spec["E"], spec["freq"], spec["dir"]
    order = np.argsort(d)
    E, d = E[:, :, order], d[order]
    dth = np.gradient(d)
    Ef = np.nansum(E * dth, axis=2)                                  # 1-D spectrum, m²/Hz
    m0 = np.trapezoid(Ef, f, axis=1)
    tp = 1 / f[np.argmax(Ef, axis=1)]
    Ed = np.trapezoid(E, f, axis=1)                                  # directional distribution
    rad = np.deg2rad(d)
    direc = np.rad2deg(np.arctan2(np.nansum(Ed * np.sin(rad) * dth, 1), np.nansum(Ed * np.cos(rad) * dth, 1))) % 360
    out = pd.DataFrame({"x": spec["x"], "y": spec["y"], "hs": 4 * np.sqrt(m0), "tp": tp, "dir": direc})
    out.loc[~(m0 > 0), ["tp", "dir"]] = np.nan
    return out


# ------------------------------------------------------------------ projects
def _cases(cfg, cases=None):
    root = cfg.results_dir
    names = sorted(d.name for d in root.iterdir() if (d / "done.json").exists()) if root.exists() else []
    return [c for c in names if c in set(cases)] if cases else names


def read_points(cfg, cases=None, grids=None):
    """Point results of finished cases as one long DataFrame.

    Columns: ``case, state, variant, grid, point`` + output quantities + the
    sea state of the case (``Hs, Tp, Dir…``).
    """
    pts = pd.read_csv(cfg.cases_dir / "_common" / "points.csv")
    q = cfg["outputs"]["point_quantities"]
    rows = []
    for c in _cases(cfg, cases):
        meta = json.loads((cfg.results_dir / c / "case.json").read_text())
        for g in meta["stages"]:
            g = g["grid"]
            f = cfg.results_dir / c / f"points_{g}.tab"
            if (grids and g not in grids) or not f.exists():
                continue
            t = read_table(f, q)
            if len(t) != len(pts):
                raise ValueError(f"{f}: {len(t)} rows but {len(pts)} output points (_common/points.csv)")
            t.insert(0, "point", pts["name"].to_numpy())
            t.insert(0, "grid", g)
            t.insert(0, "variant", meta["variant"])
            t.insert(0, "state", meta["state"])
            t.insert(0, "case", c)
            for k_, v in meta["values"].items():
                if k_ != "case":
                    t[k_] = v
            rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def read_map(cfg, case, grid, folder=None):
    """Map output (``map_<grid>.tab``) of a case as a DataFrame (one row per node).

    Looked for in ``folder`` (if given), then ``results_dir/<case>`` and ``archive_dir/<case>``.
    """
    m = cfg["outputs"]["maps"][grid]
    name = f"map_{grid}.tab"
    places = [Path(folder)] if folder else []
    places += [cfg.results_dir / case] + ([cfg.archive_dir / case] if cfg.archive_dir else [])
    for p in places:
        if (p / name).exists():
            return read_table(p / name, m["quantities"])
    raise FileNotFoundError(f"{name} of {case} not found in {[str(p) for p in places]}")


def read_spectra_case(cfg, case, grid):
    """Point spectra of a case and grid (``spec_<grid>.sp2``)."""
    return read_spectra(cfg.results_dir / case / f"spec_{grid}.sp2")


def summary(cfg, cases=None):
    """Convergence and errors of every finished case and grid (from ``PRINT_<grid>``)."""
    rows = []
    for c in _cases(cfg, cases):
        meta = json.loads((cfg.results_dir / c / "case.json").read_text())
        done = json.loads((cfg.results_dir / c / "done.json").read_text())
        for s in meta["stages"]:
            g = s["grid"]
            p = cfg.results_dir / c / f"PRINT_{g}"
            d = read_print(p) if p.exists() else {}
            d.pop("messages", None)
            rows.append({"case": c, "state": meta["state"], "variant": meta["variant"], "grid": g,
                         "minutes": done.get(f"min_{g}"), **d})
    return pd.DataFrame(rows)
