"""Delft3D outputs: raw ASCII exports and DELFT3D-WAVE NEFIS ``.dat`` map files."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from environmentaltools.common import save


def raw_files_point(
    point: list,
    mesh_filename: str,
    folder: str,
    variables: list,
    num_cases: int,
    filename: str = "seastates"
):
    """Extract time series data at a specific point from Delft3D model outputs.

    Reads Delft3D raw output files and extracts time series at the nearest
    grid point to specified coordinates. Saves results to compressed CSV.

    Args:
        point (list): [x, y] coordinates of extraction point.
        mesh_filename (str): Path to Delft3D mesh file containing grid coordinates.
        folder (str): Directory containing case subdirectories with model outputs.
        variables (list): Variable names to extract (e.g., ['hs', 'tp', 'eta']).
        num_cases (int): Number of model cases to process.
        filename (str): Output filename prefix (a trailing ``_`` is ignored).
            Defaults to "seastates".

    Returns:
        None: Saves extracted data to ZIP file with format:
            {filename}_{point[0]}_{point[1]}.zip
    """
    cases = np.arange(1, num_cases + 1)

    # Read mesh file and parse coordinate data
    fid = open(mesh_filename, "r")
    data = fid.readlines()
    readed, kline = [], -1

    # Extract coordinate lines starting at line 8
    for i in range(8, len(data)):
        if data[i].startswith(" ETA=    1 "):
            readed.append(data[i])
            kline += 1
        else:
            readed[kline] += data[i]

    # Regular expression to extract numeric values (including scientific notation)
    numeric_const_pattern = (
        r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    # Extract x and y coordinates from first two coordinate lines
    x, y = rx.findall(readed[0]), rx.findall(readed[1])

    # Convert string coordinates to floats
    for i, j in enumerate(x):
        x[i], y[i] = float(x[i]), float(y[i])

    # Determine grid structure from coordinate patterns
    idx = np.where(np.isclose(x, 2))[0][0]
    nlen = int(len(x) / idx)
    idxs = np.arange(0, len(x), idx, dtype=int)

    # Remove separator values
    for i in idxs[::-1]:
        del x[i], y[i]

    # Reshape coordinates into 2D grid
    x, y = np.reshape(np.array(x), (nlen, idx - 1)), np.reshape(
        np.array(y), (nlen, idx - 1)
    )

    # Find nearest grid point to requested location
    ids = np.where(
        np.min(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2))
        == np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
    )

    # If water level (eta) requested, find nearest point in trim file grid
    if "eta" in variables:
        datax = xr.open_mfdataset(
            folder + "/case0001/trim-guad.nc", combine="by_coords"
        )
        x = datax.XCOR.compute().data
        y = datax.YCOR.compute().data

        ids_trim = np.where(
            np.min(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2))
            == np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        )

    # Initialize output DataFrame
    data = pd.DataFrame(-1, index=cases, columns=[variables])
    
    # Loop through all cases and extract data at point
    for i in cases:
        # Read first variable file to get grid dimensions
        fid = open(folder + "/case" + str(i).zfill(4) + "/" + variables[0] + ".txt", "r")
        info = fid.readlines()
        nodesxt, nodesy, nodest = [int(nodes) for nodes in rx.findall(info[3])]
        nodesx = int(nodesxt / nodest)

        # Extract each requested variable
        for var_ in variables:
            if var_ == "eta":
                # Water level requires NetCDF trim file
                datax = xr.open_mfdataset(
                    folder + "/case" + str(i).zfill(4) + "/trim-guad.nc",
                    combine="by_coords",
                )
                z = datax.S1.compute().data  # S1 = water level
                z = z[-1, :, :]  # Extract last time step
                data.loc[i, "eta"] = z[ids_trim]
            else:
                # Other variables from text files
                data.loc[i, var_] = np.loadtxt(
                    folder + "/case" + str(i).zfill(4) + "/" + var_ + ".txt",
                    skiprows=nodesxt - nodesx + 4,
                )[ids[1][0], ids[0][0]]

    # Save extracted time series to compressed CSV
    save.to_csv(data, filename.rstrip("_") + "_" + str(point[0]) + "_" + str(point[1]) + ".zip")
    return


def raw_files(folder: str, variables: dict, case_id: str):
    """Load Delft3D raw output files for a specific case.

    Reads multiple variable output files from Delft3D model for a given case,
    handling both communication (vars_com_guad) and wave (vars_wavm) variables.

    Args:
        folder (str): Path to directory containing case subdirectories.
        variables (dict): Dictionary with keys 'vars_com_guad' and/or 'vars_wavm',
            each containing list of variable names to load.
        case_id (str): Case identifier (e.g., 'case0001').

    Returns:
        dict: Dictionary where keys are variable names and values are numpy
            arrays containing the spatial data for each variable.
    """
    # Regular expression to extract numeric values (including scientific notation)
    numeric_const_pattern = (
        r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    dic = {}
    for var_ in variables:
        if var_ == "vars_com_guad":
            # Load communication variables (COM module outputs)
            fid = open(folder / f"{case_id}" / f"{variables['vars_com_guad'][0]}.txt", "r")
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt / nodest)
            for j in variables["vars_com_guad"]:
                dic[str(j)] = np.loadtxt(
                    folder / f"{case_id}" / f"{j}.txt", skiprows=nodesxt - nodesx + 4
                )
        else:
            # Load wave variables (WAVE module outputs)
            fid = open(folder / f"{case_id}" / f"{variables['vars_wavm'][0]}.txt", "r")
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt / nodest)
            for j in variables["vars_wavm"]:
                dic[str(j)] = np.loadtxt(
                    folder / f"{case_id}" / f"{j}.txt", skiprows=nodesxt - nodesx + 4
                )

    return dic


# ---------------------------------------------------------------------------
# DELFT3D-WAVE NEFIS binary extraction (discovered by reverse-engineering)
# ---------------------------------------------------------------------------

# Variable index in the NEFIS .dat binary.
# Each entry: (group, j) where:
#   group 1 → offset = hs_start + j * step            (blocks 0–16)
#   group 2 → offset = hs_start + 17*step + j*step    (blocks 17–28)
#   group 3 → offset = hs_start + 29*step + j*step    (blocks 29–30)
#
# hs_start = file_size - 31*step  (31 float32 blocks at end of NEFIS file).
#
# Block layout verified by active-cell byte-scan against the NEFIS .def file
# (DELFT3D-WAVE NEFIS 5.00, wavm-guad-Alboran_int).  The .def element order
# for map-series is: TIME CODE HSIGN DIR PDIR PERIOD RTP DEPTH VELOC-X
# VELOC-Y TRANSP-X TRANSP-Y DSPR DISSIP LEAK QB XP YP UBOT STEEPW WLENGTH
# TPS TM02 TMM10 DHSIGN DRTM01 SETUP FX FY TP.  TIME is stored outside the
# 31-block region (part of the NEFIS header); all remaining elements map to
# binary block j = (.def index) − 1:
#
#   j=0:  CODE       j=1:  HSIGN      j=2:  DIR       j=3:  PDIR
#   j=4:  PERIOD     j=5:  RTP        j=6:  DEPTH      j=7:  VELOC-X (zeros)
#   j=8:  VELOC-Y    j=9:  TRANSP-X   j=10: TRANSP-Y   j=11: DSPR
#   j=12: DISSIP     j=13: LEAK       j=14: QB          j=15: XP
#   j=16: YP         j=17: UBOT       j=18: STEEPW      j=19: WLENGTH
#   j=20: TPS        j=21: TM02       j=22: TMM10       j=23: DHSIGN (skip)
#   j=24: DRTM01     j=25: SETUP      j=26: FX          j=27: FY
#   j=28: TP         j=29: WINDU      j=30: WINDV
_DELFT_VAR_INDEX = {
    # --- Group 1 ---
    "hsign":    (1,  1),   # block  1:  0–5.1 m, 95 808 active cells ✓
    "dir":      (1,  2),   # block  2:  8–360°  ✓
    "pdir":     (1,  3),   # block  3:  5–255°
    "period":   (1,  4),   # block  4:  0.7–15 s ✓
    "rtp":      (1,  5),   # block  5
    "depth":    (1,  6),   # block  6:  0–547 m ✓
    "veloc-x":  (1,  7),   # block  7:  all 0 (no current forcing) ✓
    "veloc-y":  (1,  8),   # block  8:  all 0 (no current forcing) ✓
    "transp-x": (1,  9),   # block  9
    "transp-y": (1, 10),   # block 10
    "dspr":     (1, 11),   # block 11:  0–77°  ✓
    "dissip":   (1, 12),   # block 12
    "leak":     (1, 13),   # block 13
    "qb":       (1, 14),   # block 14
    # --- Group 2 (starts at block 17 = hs_start + 17*step) ---
    "ubot":     (2,  0),   # block 17:  0–2.43 m/s ✓
    "steepw":   (2,  1),   # block 18
    "wlength":  (2,  2),   # block 19
    "tps":      (2,  3),   # block 20:  1–15.5 s ✓
    "tm02":     (2,  4),   # block 21:  0.6–14.7 s ✓
    "tmm10":    (2,  5),   # block 22:  0–14.9 s ✓
    # j=6 → block 23 is DHSIGN (internal, not exposed)
    "drtm01":   (2,  7),   # block 24
    "setup":    (2,  8),   # block 25:  –2.84–0.71 m ✓
    "fx":       (2,  9),   # block 26 ✓
    "fy":       (2, 10),   # block 27 ✓
    # --- Group 3 (starts at block 29 = hs_start + 29*step) ---
    "windu":    (3,  0),   # block 29:  confirmed ✓
    "windv":    (3,  1),   # block 30:  confirmed ✓
}

# NEFIS binary layout constants (DELFT3D-WAVE NEFIS 5.00 map file).
#
# File structure:
#   Bytes 0 .. hs_start-1 : NEFIS header / table structure
#   Bytes hs_start .. end  : 31 contiguous float32 blocks, each of size
#                            nmax * mmax * 4 bytes.
#   hs_start = file_size - 31 * (nmax * mmax * 4)
_NEFIS_FLOAT_BLOCKS = 31
_NEFIS_FILL         = -9000.0


def _delft_grid_params(case_dir, dat_name, grd_name):
    """Read grid dimensions from .grd and compute hs_start + M-direction roll.

    The NEFIS binary stores each N-row with a circular shift in the M-direction
    relative to the .grd ordering: row n of the flat array starts at some M
    offset M_start instead of M=0, wraps at M=mmax-1, then continues from M=0.
    This function determines the shift (m_roll) so callers can correct for it
    with np.roll(arr, -m_roll, axis=1).
    """
    grd_path = case_dir / grd_name
    dat_path = case_dir / dat_name

    # Read mmax, nmax AND the first ETA row (N=0) of X coordinates from .grd
    mmax = nmax = None
    current = []
    in_eta = False
    x_row0 = []

    with open(grd_path, encoding="latin-1") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            if "Missing" in s or "Coordinate" in s:
                continue
            if mmax is None and "=" not in s:
                parts = s.split()
                if len(parts) == 2:
                    try:
                        mmax, nmax = int(parts[0]), int(parts[1])
                    except ValueError:
                        pass
                continue
            if mmax is None:
                continue
            if s == "0 0 0":
                continue
            if s.startswith("ETA="):
                if in_eta:
                    # Second ETA= reached: first row is complete
                    x_row0 = current
                    break
                in_eta = True
                current = [float(v) for v in s.split()[2:]]
            elif in_eta:
                current.extend(float(v) for v in s.split())
                if len(current) >= mmax:
                    x_row0 = current[:mmax]
                    break

    if mmax is None:
        raise ValueError(f"Cannot read grid dimensions from {grd_path}")

    step      = nmax * mmax * 4
    file_size = dat_path.stat().st_size
    hs_start  = file_size - _NEFIS_FLOAT_BLOCKS * step

    # Validate hs_start; if the formula gives an invalid position (e.g. when
    # the file was written with a different number of float blocks), scan
    # candidate block-counts in descending order and pick the first one whose
    # XP block (j=15) contains plausible UTM easting values.
    def _xp_is_valid(hs_candidate):
        """Return True if block 15 from hs_candidate looks like UTM eastings."""
        xp_off = hs_candidate + 15 * step
        if hs_candidate < 0 or xp_off + mmax * 4 > file_size:
            return False
        with open(dat_path, "rb") as _f:
            _f.seek(xp_off)
            row = np.frombuffer(_f.read(mmax * 4), dtype="<f4")
        valid_frac = ((row > 100_000) & (row < 900_000)).mean()
        return valid_frac > 0.05          # ≥5 % of first row in UTM-east range

    if not _xp_is_valid(hs_start):
        # Try nearby block counts (±6 around the canonical 31)
        found = False
        for nblocks in range(_NEFIS_FLOAT_BLOCKS + 6, _NEFIS_FLOAT_BLOCKS - 7, -1):
            candidate = file_size - nblocks * step
            if _xp_is_valid(candidate):
                hs_start = candidate
                found = True
                break
        if not found:
            approx_blocks = file_size / step if step else 0
            hint = (
                f" The file can hold only ~{approx_blocks:.1f} blocks of that "
                f"size, which strongly suggests that '{dat_path.name}' and the "
                f"supplied .grd file (giving {nmax}×{mmax}) belong to "
                "different grids — check that dat_name and grd_name refer to "
                "the same grid (e.g. both 'int' or both 'ext')."
                if approx_blocks < 15 else
                " Tried all block counts in "
                f"{_NEFIS_FLOAT_BLOCKS-6}–{_NEFIS_FLOAT_BLOCKS+6} — none "
                "produced a valid XP block."
            )
            raise ValueError(
                f"Cannot locate float-data region in {dat_path.name}: "
                f"file_size={file_size} B, step={step} B "
                f"({nmax}×{mmax}×4).{hint}"
            )

    # Compute M-direction roll: find the circular shift between .dat and .grd.
    # The first valid X value in .grd row 0 is the M=M_v reference; locate its
    # position in the first row of XP from the .dat to get the roll amount.
    m_roll = 0
    if len(x_row0) == mmax:
        x_arr  = np.array(x_row0, dtype=float)
        valid  = (x_arr > 100_000) & (x_arr < 800_000)
        if valid.any():
            m_v   = int(np.where(valid)[0][0])   # first valid M index in .grd
            x_ref = float(x_arr[m_v])
            with open(dat_path, "rb") as f:
                f.seek(hs_start + 15 * step)     # XP block
                xp_row0 = np.frombuffer(f.read(mmax * 4), dtype="<f4").copy().astype(float)
            valid_xp = (xp_row0 > 100_000) & (xp_row0 < 800_000)
            if valid_xp.any():
                k      = int(np.argmin(np.abs(xp_row0 - x_ref)))
                m_roll = (k - m_v) % mmax

    return {"nmax": nmax, "mmax": mmax, "hs_start": hs_start, "m_roll": m_roll}


def _delft_var_offset(gp, group, j):
    hs   = gp["hs_start"]
    step = gp["nmax"] * gp["mmax"] * 4
    if group == 1:
        return hs + j * step                  # blocks 0-13
    if group == 2:
        return hs + 17 * step + j * step      # blocks 17-27 (XP/YP at 15-16)
    return hs + 29 * step + j * step          # blocks 29-30 (WIND)


def _delft_read_var(dat_path, varname, gp, fill_threshold):
    """Read a 2-D float32 array from a NEFIS .dat file."""
    group, j = _DELFT_VAR_INDEX[varname]
    offset   = _delft_var_offset(gp, group, j)
    npts     = gp["nmax"] * gp["mmax"]
    with open(dat_path, "rb") as f:
        f.seek(offset)
        arr = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
    arr[arr < fill_threshold] = np.nan
    arr = arr.reshape(gp["nmax"], gp["mmax"])
    if gp.get("m_roll", 0):
        arr = np.roll(arr, -gp["m_roll"], axis=1)
    return arr


def _delft_read_coords(dat_path, gp, fill_threshold):
    """Read XP, YP coordinates and build active-cell mask.

    XP is at block 15 and YP at block 16 from hs_start in the NEFIS 5.00
    file layout (verified by byte-scan: 98 % of cells fall in UTM easting
    range 291 848 – 342 104, consistent with Alboran Sea UTM Zone 30N).
    """
    step  = gp["nmax"] * gp["mmax"] * 4
    npts  = gp["nmax"] * gp["mmax"]
    start = gp["hs_start"] + 15 * step   # XP at block 15, YP at block 16
    with open(dat_path, "rb") as f:
        f.seek(start)
        x = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
        y = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
    x = x.reshape(gp["nmax"], gp["mmax"])
    y = y.reshape(gp["nmax"], gp["mmax"])
    if gp.get("m_roll", 0):
        x = np.roll(x, -gp["m_roll"], axis=1)
        y = np.roll(y, -gp["m_roll"], axis=1)
    # Active cells have valid UTM coordinates; inactive (land/outside) cells
    # are stored as 0.0 in the NEFIS output.
    active = (x > 0) & (x < 1e6) & (y > 1e6)
    return x, y, active


def extract_wave_dat(
    cases_dir,
    dat_name,
    grd_name,
    points,
    variables,
    method="nearest",
    case_pattern="caso_*",
    point_label_col=None,
    input_df=None,
    fill_threshold=_NEFIS_FILL,
    output=None,
    verbose=True,
):
    """Extract DELFT3D-WAVE variables from NEFIS binary (.dat) files at given points.

    Reads the NEFIS binary output of DELFT3D-WAVE (one ``.dat`` file per model
    case) and extracts the requested wave variables at a set of spatial points
    across all cases found in *cases_dir*. The binary layout is decoded by
    reverse-engineering (see ``_DELFT_VAR_INDEX``).

    Grid dimensions are derived automatically from the ``.grd`` ASCII file,
    avoiding any need to hard-code ``nmax`` / ``mmax``.

    Args:
        cases_dir (str | Path): Root directory that contains the case
            sub-directories (e.g. ``caso_001``, ``caso_002`` …).
        dat_name (str): Name of the ``.dat`` file inside each case directory
            (e.g. ``"wavm-guad-Alboran_int.dat"``). Cases that do not
            contain this file are silently skipped.
        grd_name (str): Name of the ``.grd`` ASCII grid file inside the first
            valid case directory (e.g. ``"Alboran_int.grd"``).
        points (pd.DataFrame): Table of extraction points with at least
            ``x`` and ``y`` columns in the same UTM coordinate system as
            the model grid. The caller is responsible for projecting from
            lat/lon if necessary (use ``pyproj.Transformer``).
        variables (list[str]): Wave variables to extract. Available names::

                hsign, dir, pdir, period, rtp, depth,
                veloc-x, veloc-y, transp-x, transp-y,
                dspr, dissip, leak, qb,
                ubot, steepw, wlength, tps, tm02, tmm10,
                drtm01, setup, fx, fy,
                windu, windv

        method (str): Extraction method. ``"nearest"`` uses the closest
            active grid cell; ``"interpolate"`` uses bilinear interpolation
            over the surrounding active cells. Defaults to ``"nearest"``.
        case_pattern (str): Glob pattern to discover case sub-directories.
            Defaults to ``"caso_*"``.
        point_label_col (str | None): Column in *points* to use as point
            identifier in the output columns (e.g. ``"id"``). If ``None``,
            the DataFrame index is used. Defaults to ``None``.
        input_df (pd.DataFrame | None): Optional case metadata indexed by
            ``case_id`` (integer). When provided, its columns are prepended
            to the output with the prefix ``in_``. Defaults to ``None``.
        fill_threshold (float): Values below this threshold are treated as
            fill / no-data and set to ``NaN``. Defaults to ``-9000.0``.
        output (str | Path | None): If given, saves the result to an Excel
            file with one sheet per extraction point. Defaults to ``None``.
        verbose (bool): Print progress messages. Defaults to ``True``.

    Returns:
        pd.DataFrame: One row per case with columns:

            * ``caso_id`` — integer case identifier
            * ``in_<col>`` — columns from *input_df* (if provided)
            * ``<label>_<var>`` — extracted value for each point × variable

    Raises:
        FileNotFoundError: If no case directories containing *dat_name* are
            found under *cases_dir*.
        ValueError: If *variables* contains an unknown variable name or if
            grid dimensions cannot be read from the ``.grd`` file.
        KeyError: If *point_label_col* is not a column of *points*.

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from pyproj import Transformer
        >>> from environmentaltools.models import delft3d
        >>>
        >>> # Project points from lat/lon to UTM 30N
        >>> df_pts = pd.read_csv("registro_puntos.csv")
        >>> tr = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
        >>> df_pts["x"], df_pts["y"] = tr.transform(df_pts["longitude"], df_pts["latitude"])
        >>>
        >>> # Load case metadata
        >>> meta = pd.read_csv("500_cases.csv").set_index("id")
        >>>
        >>> df = delft3d.extract_wave_dat(
        ...     cases_dir=Path("SALIDAS_DELFT/500_casos_Alboran"),
        ...     dat_name="wavm-guad-Alboran_int.dat",
        ...     grd_name="Alboran_int.grd",
        ...     points=df_pts,
        ...     variables=["hsign", "dir", "period", "dspr"],
        ...     point_label_col="id",
        ...     input_df=meta,
        ...     output="extraccion.xlsx",
        ... )
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata

    cases_dir = Path(cases_dir)

    # ------------------------------------------------------------------
    # Validate variables
    # ------------------------------------------------------------------
    unknown = [v for v in variables if v not in _DELFT_VAR_INDEX]
    if unknown:
        raise ValueError(
            f"Unknown variable(s): {unknown}. "
            f"Available: {sorted(_DELFT_VAR_INDEX)}"
        )

    # ------------------------------------------------------------------
    # Discover case directories
    # ------------------------------------------------------------------
    case_dirs = sorted(cases_dir.glob(case_pattern))
    case_dirs = [d for d in case_dirs if (d / dat_name).exists()]
    if not case_dirs:
        raise FileNotFoundError(
            f"No case directories matching '{case_pattern}' with '{dat_name}' "
            f"found under {cases_dir}"
        )

    # ------------------------------------------------------------------
    # Grid parameters (from first valid case)
    # ------------------------------------------------------------------
    gp = _delft_grid_params(case_dirs[0], dat_name, grd_name)
    dat0 = case_dirs[0] / dat_name
    x_grid, y_grid, active = _delft_read_coords(dat0, gp, fill_threshold)

    if verbose:
        print(f"Cases found        : {len(case_dirs)}")
        print(f"Grid dims          : {gp['nmax']} × {gp['mmax']}")
        print(f"Active cells       : {active.sum()}")
        print(f"Variables          : {variables}")
        print(f"Method             : {method}")

    # ------------------------------------------------------------------
    # Build KD-tree over active cells
    # ------------------------------------------------------------------
    xy_active = np.column_stack([x_grid[active], y_grid[active]])
    tree      = cKDTree(xy_active)
    active_rc = np.argwhere(active)   # (N_active, 2) → row/col indices

    # ------------------------------------------------------------------
    # Resolve point labels
    # ------------------------------------------------------------------
    if point_label_col is not None:
        labels_pts = points[point_label_col].astype(str).tolist()
    else:
        labels_pts = [str(i) for i in points.index]

    pts_xy = points[["x", "y"]].values

    # ------------------------------------------------------------------
    # Check which points fall inside the active domain
    # Criterion: nearest active cell closer than 2.5 × grid resolution
    # ------------------------------------------------------------------
    sample_n  = min(200, len(xy_active))
    rng       = np.random.default_rng(0)
    sample_i  = rng.choice(len(xy_active), size=sample_n, replace=False)
    d2nd, _   = tree.query(xy_active[sample_i], k=2)
    resolution = np.median(d2nd[:, 1])
    threshold  = resolution * 2.5

    inside_mask = []
    nearest_rc  = {}
    for label, (xp, yp) in zip(labels_pts, pts_xy):
        dist, ii = tree.query([xp, yp])
        ok = dist <= threshold
        inside_mask.append(ok)
        if ok:
            nearest_rc[label] = tuple(active_rc[ii])

    pts_inside = [
        (lbl, xp, yp)
        for lbl, (xp, yp), ok in zip(labels_pts, pts_xy, inside_mask)
        if ok
    ]
    pts_outside = [lbl for lbl, ok in zip(labels_pts, inside_mask) if not ok]

    if verbose:
        print(f"Grid resolution    : {resolution:.0f} m  (threshold {threshold:.0f} m)")
        print(f"Points inside grid : {len(pts_inside)}")
        if pts_outside:
            print(f"Points outside     : {pts_outside}  (skipped)")

    if not pts_inside:
        raise ValueError(
            "No extraction points fall within the active model domain. "
            "Check that 'x' and 'y' in `points` are in the grid's CRS."
        )

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------
    rows = []
    for i, case_dir in enumerate(case_dirs):
        case_id  = int(case_dir.name.split("_")[1])
        dat_path = case_dir / dat_name

        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"  Case {case_id:04d}  ({i + 1}/{len(case_dirs)})")

        row = {"caso_id": case_id}

        # Merge input metadata
        if input_df is not None and case_id in input_df.index:
            for col in input_df.columns:
                row[f"in_{col}"] = input_df.loc[case_id, col]

        # Extract each variable at each point
        for label, xp, yp in pts_inside:
            for var in variables:
                try:
                    arr = _delft_read_var(dat_path, var, gp, fill_threshold)
                    if method == "nearest":
                        r, c = nearest_rc[label]
                        val  = float(arr[r, c])
                    else:
                        mask = ~np.isnan(arr) & active
                        if mask.any():
                            val = float(griddata(
                                points=np.column_stack([x_grid[mask], y_grid[mask]]),
                                values=arr[mask],
                                xi=[[xp, yp]],
                                method="linear",
                            )[0])
                        else:
                            val = np.nan
                except Exception:
                    val = np.nan
                row[f"{label}_{var}"] = val

        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("caso_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optional Excel output (one sheet per point)
    # ------------------------------------------------------------------
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        input_cols = [c for c in df_out.columns
                      if c == "caso_id" or c.startswith("in_")]
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for label, _, _ in pts_inside:
                pt_cols = [c for c in df_out.columns if c.startswith(f"{label}_")]
                df_pt   = df_out[input_cols + pt_cols].copy()
                df_pt.columns = [
                    c.replace(f"{label}_", "") if c.startswith(f"{label}_") else c
                    for c in df_pt.columns
                ]
                df_pt.to_excel(writer, sheet_name=str(label), index=False)
        if verbose:
            print(f"\nSaved to: {output.resolve()}")

    return df_out
