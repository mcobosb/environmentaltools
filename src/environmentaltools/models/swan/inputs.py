"""Writers of SWAN input files: grid/bottom files, nesting points and the command file (INPUT)."""

from pathlib import Path

import numpy as np

from ..common.grids import MISSING

SIDES = {"N", "NW", "W", "SW", "S", "SE", "E", "NE"}


# ----------------------------------------------------------------- data files
def write_grid_files(grid, folder, depmin=0.05):
    """Write the coordinate (curvilinear grids only) and bottom files of a grid.

    Coordinates follow the Delft3D ``TMP_grid2swan`` layout (``x-coordinates``
    header + all x, ``y-coordinates`` header + all y; i fastest), read with
    ``READGRID COORDINATES 1. '<file>' 4 0 1 FREE``. Depths at the nodes are
    limited below by ``depmin`` (dry nodes), as Delft3D-WAVE does.

    Returns:
        dict: ``{"coordinates": name | None, "bottom": name}`` (file names in ``folder``).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {"coordinates": None, "bottom": f"bottom_{grid.name}.txt"}
    if grid.kind == "curvilinear":
        files["coordinates"] = f"coordinates_{grid.name}.txt"
        x = np.where(np.isnan(grid.x), MISSING, grid.x).ravel()
        y = np.where(np.isnan(grid.y), MISSING, grid.y).ravel()
        txt = "x-coordinates\n" + "\n".join(f"  {v:.17E}" for v in x) + \
              "\ny-coordinates\n" + "\n".join(f"  {v:.17E}" for v in y) + "\n"
        (folder / files["coordinates"]).write_text(txt)
    if grid.depth is None:
        raise ValueError(f"Grid '{grid.name}' has no bathymetry")
    d = np.maximum(np.nan_to_num(grid.depth, nan=depmin), depmin)
    (folder / files["bottom"]).write_text("\n".join(f"  {v:.6E}" for v in d.ravel()) + "\n")
    return files


def write_nest_points(grid, folder):
    """Write the perimeter nodes of a nested grid (Delft3D-WAVE order) → ``nest_<grid>.pts``."""
    j, i = grid.perimeter()
    xy = np.column_stack([grid.x[j, i], grid.y[j, i]])
    xy[np.isnan(xy)] = MISSING
    name = f"nest_{grid.name}.pts"
    np.savetxt(Path(folder) / name, xy, fmt="%16.6f")
    return name


def write_points(points, folder):
    """Write output points (DataFrame with ``x``, ``y``) → ``points.pts``."""
    name = "points.pts"
    np.savetxt(Path(folder) / name, points[["x", "y"]].to_numpy(float), fmt="%16.4f")
    return name


# ------------------------------------------------------------ command file
def _f(v, nd=2):
    return f"{v:.{nd}f}"


def _boundary_commands(cfg, grid, state):
    """BOUN SHAPE + BOUN SIDE/SEGMENT commands of a non-nested grid."""
    bs = cfg["boundary_spectrum"]
    shape = f"BOUN SHAPE {bs['shape']}" + (f" {_f(bs['gamma'])}" if bs["shape"].upper().startswith("JON") else "")
    shape += f" {bs['period']} DSPR {bs['spreading']}"
    dd = state.get("spreading")
    dd = bs["spreading_value"] if dd is None or (isinstance(dd, float) and np.isnan(dd)) else dd
    hs, tp, dr = state["Hs"], state["Tp"], state["Dir"]
    lines = [shape]
    for b in cfg["grids"][grid.name]["boundaries"]:
        if "side" in b:
            side = str(b["side"]).upper()
            if side not in SIDES:
                raise ValueError(f"Unknown side '{side}'")
            loc = f"BOUN SIDE {side} {'CCW' if b.get('ccw', True) else 'CLOCKWISE'}"
        elif "segment_ij" in b:
            loc = "BOUN SEGMENT IJ " + " ".join(str(int(v)) for v in b["segment_ij"])
        elif "segment_xy" in b:
            loc = "BOUN SEGMENT XY " + " ".join(f"{p[0]} {p[1]}" for p in b["segment_xy"])
        else:
            raise ValueError(f"Boundary of grid '{grid.name}' needs 'side', 'segment_ij' or 'segment_xy'")
        if b.get("variable"):
            pars = []
            for p in b["variable"]:
                pars.append(f"{_f(p['s'], 1)} {_f(hs * p.get('hs', 1.0))} {_f(tp * p.get('tp', 1.0))} "
                            f"{_f((dr + p.get('dir', 0.0)) % 360)} {_f(dd)}")
            lines.append(f"{loc} VARIABLE PAR " + " _\n    ".join(pars))
        else:
            lines.append(f"{loc} CONSTANT PAR {_f(hs)} {_f(tp)} {_f(dr % 360)} {_f(dd)}")
    return lines


def build_input(cfg, grid, state, files, children=(), points=False, project_nr="1"):
    """Text of the SWAN command file for one grid and one sea state.

    Args:
        cfg (SwanConfig): Configuration (possibly of a variant).
        grid (Grid): Computational grid.
        state (dict): ``Hs``, ``Tp``, ``Dir``, ``Vv``, ``DirV`` and optionally
            ``spreading`` and ``level``.
        files (dict): File names of this grid (``coordinates``, ``bottom``) and
            ``nest`` → {child grid: nest points file}.
        children (list[str]): Grids nested in this one (NESTOUT is written for them).
        points (bool): Whether ``points.pts`` exists (point output).
        project_nr (str): Run number (≤ 4 characters).

    Returns:
        str: Command file content (CRLF line endings are added when written).
    """
    s, sg, g = cfg["set"], cfg["spectral_grid"], cfg["grids"][grid.name]
    level = state.get("level")
    level = s["level"] if level is None or (isinstance(level, float) and np.isnan(level)) else level
    out = [
        "$*************************** HEADING ****************************",
        f"PROJECT '{cfg['project'][:16]}' '{str(project_nr)[:4]}'",
        f"  '{state.get('case', '')}'",
        f"  'grid {grid.name}'",
        "$*************************** MODEL INPUT ************************",
        f"SET LEVEL={_f(level)} NOR={_f(s['nor'])} DEPMIN={s['depmin']} MAXMES={s['maxmes']} "
        f"MAXERR={s['maxerr']} GRAV={s['grav']} RHO={s['rho']} INRHOG={s['inrhog']} {s['convention']}",
        "MODE STATIONARY TWODIMENSIONAL",
    ]
    if grid.spherical:
        out.append("COORDINATES SPHERICAL")
    circle = f"CIRCLE {sg['ndir']} {sg['fmin']} {sg['fmax']} {sg['msc']}"
    if grid.kind == "curvilinear":
        out += [f"CGRID CURVILINEAR {grid.mxc} {grid.myc} EXCEPTION {MISSING} {MISSING} {circle}",
                f"READGRID COORDINATES 1. '{files['coordinates']}' 4 0 1 FREE",
                f"INPGRID BOTTOM CURVILINEAR 0. 0. {grid.mxc} {grid.myc}"]
    else:
        r = grid.regular
        out += [f"CGRID REGULAR {r['xp']} {r['yp']} {r['alp']} {r['dx'] * grid.mxc} {r['dy'] * grid.myc} "
                f"{grid.mxc} {grid.myc} {circle}",
                f"INPGRID BOTTOM REGULAR {r['xp']} {r['yp']} {r['alp']} {grid.mxc} {grid.myc} {r['dx']} {r['dy']}"]
    out.append(f"READINP BOTTOM 1.0 '{files['bottom']}' 4 0 FREE")
    vv = state.get("Vv")
    if vv is not None and not (isinstance(vv, float) and np.isnan(vv)):
        vv = max(float(vv), float(cfg["states"]["min_wind"] or 0.0))
        out.append(f"WIND {_f(vv)} {_f(state['DirV'] % 360)}")

    out.append("$*************************** BOUNDARY ***************************")
    if g.get("nested_in"):
        out.append(f"BOUN NEST '{files['nest_in']}' CLOSED")
    else:
        out += _boundary_commands(cfg, grid, state)

    out.append("$*************************** PHYSICS ****************************")
    out += [c for c in cfg["physics"].values() if c]
    out.append("$*************************** NUMERICS ***************************")
    out += [c for c in cfg["numerics"].values() if c]

    out.append("$*************************** OUTPUT *****************************")
    for k, child in enumerate(children, 1):
        out += [f"POINTS 'NEST{k}' FILE '{files['nest'][child]}'",
                f"SPEC 'NEST{k}' SPEC2D ABS 'nest_{child}.sp2'"]
    o = cfg["outputs"]
    if points:
        out += ["POINTS 'PTS' FILE 'points.pts'",
                f"TABLE 'PTS' HEAD 'points_{grid.name}.tab' " + " ".join(o["point_quantities"])]
        if o.get("point_spectra"):
            out.append(f"SPEC 'PTS' SPEC2D ABS 'spec_{grid.name}.sp2'")
    m = (o.get("maps") or {}).get(grid.name)
    if m:
        frame = "COMPGRID"
        if m.get("subgrid"):
            i1, i2, j1, j2 = m["subgrid"]
            out.append(f"GROUP 'MAP' SUBGRID {i1} {i2} {j1} {j2}")
            frame = "MAP"
        out.append(f"TABLE '{frame}' NOHEAD 'map_{grid.name}.tab' " + " ".join(m["quantities"]))
    out += ["$*************************** COMPUTE ****************************",
            "COMPUTE STATIONARY", "STOP", ""]
    return "\n".join(out)


def write_input(path, text):
    """Write a command file with CRLF line endings (as SWAN on Windows expects from Delft3D)."""
    Path(path).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("latin-1"))
