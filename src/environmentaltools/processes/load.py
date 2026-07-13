import re

import numpy as np
import pandas as pd
from scipy.io import loadmat as ldm
import xarray as xr

from environmentaltools.common import read, save

def create_mesh_dictionary(fname, uf=None):
    """Load mesh parameters from Excel file into dictionary.

    Parameters
    ----------
    fname : str
        Path to Excel file containing mesh configuration
    uf : str, optional
        Specific worksheet/column name to extract. If None, returns entire file.

    Returns
    -------
    dict
        Dictionary with mesh parameters

    Notes
    -----
    Reads Excel file using environmentaltools.common.read.xlsx.
    If uf is specified, extracts only that column/sheet as dictionary.

    Examples
    --------
    >>> params = create_mesh_dictionary('mesh_config.xlsx')
    >>> params_sheet = create_mesh_dictionary('mesh_config.xlsx', uf='grid1')
    """
    info = read.xlsx(fname)
    if uf is not None:
        params = info[uf].to_dict()
    else:
        params = info

    return params




def read_cshore(file_, path):
    """Load CSHORE model output files.

    Parameters
    ----------
    file_ : str
        Output file type: 'bprof', 'bsusl', 'cross', 'crvol', 'energ', 'longs',
        'lovol', 'param', 'rolle', 'setup', 'swase', 'timse', 'xmome', 'xvelo',
        'ymome', 'yvelo'
    path : str
        Directory path containing CSHORE output files

    Returns
    -------
    pd.DataFrame
        DataFrame with output data, indexed by cross-shore position (meters)

    Notes
    -----
    CSHORE output file structure:
    
    - Files named as 'O' + ``file_``.upper() (e.g., 'OBPROF')
    - First row contains metadata (number of points for 'bprof')
    - Data is whitespace-delimited
    
    Variable definitions:
    - bprof: Beach profile elevation
    - bsusl: Bed load and suspended load probabilities and velocities
    - cross: Cross-shore sediment transport rates
    - energ: Energy flux and dissipation
    - longs: Longshore sediment transport rates
    - param: Wave parameters (period, bed load, sigma)
    - rolle: Roller energy flux
    - setup: Wave setup, depth, and standard deviation
    - swase: Swash zone parameters
    - timse: Time series of overtopping and transport
    - xmome: Cross-shore momentum (radiation stress, bed shear)
    - xvelo: Cross-shore velocities
    - ymome: Longshore momentum
    - yvelo: Longshore velocities

    Examples
    --------
    >>> df_profile = read_cshore('bprof', './cshore_run')
    >>> df_setup = read_cshore('setup', './cshore_run')
    """
    header = {'bprof': ["z"],
              'bsusl': [r'$P_b$', r'$P_s$', r'$V_s$'],
              'cross': [r'$Q_{b,x}$', r'$Q_{s,x}$', r'$Q_{b,x} + Q_{s,x}$'],
              'crvol': [],
              'energ': [r'Eflux (m3/s)', 'Db (m2/s)', 'Df (m2/s)'],
              'longs': [r'$Q_{b,y}$', r'$Q_{s,y}$', r'$Q_{b,y} + Q_{s,y}$'],
              'lovol': [],
              'param': ['T (s)', r'$Q_b$ (nondim)', 'Sigma* (nondim)'],
              'rolle': ['Rq (m2/s)'],
              'setup': [r'$\eta + S_{tide}$ (m)', 'd (m)', r'$\sigma_{eta}$ (m)'],
              'swase': ['de (m)', 'Uxe (m/s)', 'Qxe (m2/s)'],
              'timse': ['t (id)', 't (s)', 'q0 (m2/s)', 'qbx,lw (m2/s)', 'qsx,lw (m2/s)'],
              'xmome': ['Sxx (m2)', 'taubx (m)'],
              'xvelo': [r'$U_x$', r'$U_{x,std}$'],
              'ymome': ['Sxx (m2)', 'taubx (m)'],
              'yvelo': ['sin theta (unitary)', r'$U_y$', r'$U_{y,std}$']
              }

    # TODO: include morphology options
    # EWD: Output exceedance probability 0.015
    # q0: wave overtopping rate, qbx,lw: cross-shore bedload transport rate at the landward end of the computation domain
    filename = path + '/' + 'O' + file_.upper()
    if file_ == 'bprof':
        fid = open(filename, 'rb')
        properties = fid.readline()
        id_ = int(properties.split()[1])
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, index_col=0, names=header[file_])
        df = df.iloc[:id_, :]
    else:
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, index_col=0, names=header[file_])
    
    # Index represents cross-shore distance in meters
    df.columns = df.columns.astype("str") 

    return df


def read_copla(fname, grid=None):
    """Load COPLA model velocity field output.

    Parameters
    ----------
    fname : str
        Path to COPLA velocity output file
    grid : dict, optional
        Existing grid dictionary to update. If None, creates new dictionary.

    Returns
    -------
    dict
        Grid dictionary with keys:
        - 'u': East-west velocity component (m/s)
        - 'v': North-south velocity component (m/s)
        - 'U': Velocity magnitude (m/s)
        - 'DirU': Current direction (degrees, oceanographic convention)

    Notes
    -----
    File format:
    - Skips first 7 header rows
    - Columns: x, y, u, v (whitespace-delimited)
    - Data reshaped to 2D grid with ghost cells padding
    
    Direction convention:
    - 0° = North, 90° = East (oceanographic)
    - Computed from arctan2(v, u) + 90°

    Examples
    --------
    >>> grid = read_copla('velocity.001')
    >>> print(grid['U'].shape)
    >>> print(f"Max velocity: {grid['U'].max():.2f} m/s")
    """
    data = pd.read_csv(fname, skiprows=7, delim_whitespace=True, header=None, index_col=0, names=['x', 'y', 'u', 'v'])
    _, x = np.meshgrid(data.y.unique(), data.x.unique())

    if grid is None:
        grid = {}

    grid = dict()
    nx, ny = np.shape(x)
    for var_ in ['u', 'v']:
        # Create arrays with ghost cell padding (nx+2, ny+2)
        grid[var_] = np.zeros([nx+2, ny+2])
        grid[var_][1:-1, 1:-1] = data[var_].to_numpy().reshape([nx, ny])
    
    # Compute velocity magnitude and direction
    grid['U'] = np.sqrt(grid['u']**2 + grid['v']**2)
    grid['DirU'] = np.fmod(np.rad2deg(np.arctan2(grid['v'], grid['u'])) + 90, 360)

    return grid


def read_swan(fname, grid=None, vars_=None):
    """Load SWAN wave model output from MATLAB file.

    Parameters
    ----------
    fname : str
        Path to SWAN .mat output file
    grid : dict, optional
        Existing grid dictionary to update. If None, creates new dictionary.
    vars_ : list of str, optional
        Variable names for output. Default: ['x', 'y', 'depth', 'Qb', 'L', 
        'Setup', 'Hs', 'DirM']

    Returns
    -------
    dict
        Grid dictionary containing:
        - 'x': X coordinates (m)
        - 'y': Y coordinates (m)
        - 'depth': Water depth (m)
        - 'Qb': Wave energy dissipation (W/m²)
        - 'L': Wavelength (m)
        - 'Setup': Wave setup (m)
        - 'Hs': Significant wave height (m)
        - 'DirM': Mean wave direction (degrees)
        - 'kp': Peak wave number (rad/m), computed as 2π/L

    Notes
    -----
    - Reads MATLAB file with variables: Xp, Yp, Depth, Qb, Wlen, Setup, Hsig, Dir
    - NaN values replaced with 1e-6 for numerical stability
    - Wave number computed from wavelength: kp = 2π/L

    Examples
    --------
    >>> wave_grid = swan('swan_output.mat')
    >>> print(f"Max Hs: {wave_grid['Hs'].max():.2f} m")
    >>> print(f"Mean direction: {wave_grid['DirM'].mean():.1f}°")
    """
    if not vars_:
        vars_ = ['x', 'y', 'depth', 'Qb', 'L', 'Setup', 'Hs', 'DirM']
    
    if grid is None:
        grid = {}

    # Load MATLAB file
    swan_dictionary = ldm(fname)
    
    # Map SWAN variable names to output names and replace NaN with small value
    for ind_, var_ in enumerate(['Xp', 'Yp', 'Depth', 'Qb', 'Wlen', 'Setup', 'Hsig', 'Dir']):          
        grid[vars_[ind_]] = swan_dictionary[var_]
        grid[vars_[ind_]][np.isnan(grid[vars_[ind_]])] = 1e-6

    # Compute wave number from wavelength
    grid['kp'] = 2*np.pi/grid['L'] 

    return grid



def delft_raw_files_point(point, mesh_filename, folder, vars_, nocases, filename='seastates_'):
    """Extract time series at specific point from Delft3D model outputs.

    Parameters
    ----------
    point : tuple or list
        (x, y) coordinates of extraction point
    mesh_filename : str
        Path to Delft3D mesh file for coordinate mapping
    folder : str
        Directory containing case subdirectories (case0001, case0002, etc.)
    vars_ : list of str
        Variables to extract (e.g., ['hs', 'tp', 'eta'])
    nocases : int
        Number of cases to process
    filename : str, optional
        Output filename prefix. Default: 'seastates\\_'

    Returns
    -------
    None
        Saves extracted data to CSV file: {filename}{x}_{y}.zip

    Notes
    -----
    File structure expected:
    - folder/case####/var.txt for most variables
    - folder/case####/trim-guad.nc for 'eta' (water level)
    
    Algorithm:
    1. Parse mesh file to extract coordinates
    2. Find nearest grid point to requested location
    3. Extract all variables at that point for all cases
    4. Save to compressed CSV file
    
    Special handling for 'eta':
    - Reads from NetCDF file (trim-guad.nc)
    - Uses last time step: z[-1, :, :]
    - Different coordinate system than other variables

    Examples
    --------
    >>> delft_raw_files_point(
    ...     point=(430000, 4500000),
    ...     mesh_filename='mesh.dat',
    ...     folder='./delft_runs',
    ...     vars_=['hs', 'tp', 'dir'],
    ...     nocases=100
    ... )
    """
    cases = np.arange(1, nocases+1)

    # Parse mesh file to extract coordinates
    fid = open(mesh_filename, 'r')
    data = fid.readlines()
    readed, kline = [], -1

    # Combine multi-line coordinate entries
    for i in range(8, len(data)):
        if data[i].startswith(' ETA=    1 '):
            readed.append(data[i])
            kline += 1
        else:
            readed[kline] += data[i]

    # Extract numeric values using regex
    numeric_const_pattern = r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    x, y = rx.findall(readed[0]), rx.findall(readed[1])

    # Convert to float
    for i, j in enumerate(x):
        x[i], y[i] = float(x[i]), float(y[i])

    # Reshape coordinates to 2D grid
    idx = np.where(np.isclose(x, 2))[0][0]
    nlen = int(len(x)/idx)
    idxs = np.arange(0, len(x), idx, dtype=int)

    # Remove boundary points
    for i in idxs[::-1]:
        del x[i], y[i]

    x, y = np.reshape(np.array(x), (nlen, idx-1)), np.reshape(np.array(y), (nlen, idx-1))
    
    # Find nearest grid point to requested location
    ids = np.where(np.min(np.sqrt((x - point[0])**2 + (y - point[1])**2)) == np.sqrt((x - point[0])**2 + (y - point[1])**2))

    # Special handling for water level (eta) - uses different coordinate system
    if 'eta' in vars_:
        datax = xr.open_mfdataset(folder + '/case0001/trim-guad.nc', combine='by_coords')
        x = datax.XCOR.compute().data
        y = datax.YCOR.compute().data

        ids_trim = np.where(np.min(np.sqrt((x - point[0])**2 + (y - point[1])**2)) == np.sqrt((x - point[0])**2 + (y - point[1])**2))
    
    # Initialize output DataFrame
    data = pd.DataFrame(-1, index=cases, columns=[vars_])
    
    # Extract data for each case
    for i in cases:
        print(f"Processing case {i}/{nocases} for point {point}...")
        # Read header to get grid dimensions
        fid = open(folder + '/case' + str(i).zfill(4) + '/' + vars_[0] + '.txt', 'r')
        info = fid.readlines()
        nodesxt, nodesy, nodest = [int(nodes) for nodes in rx.findall(info[3])]
        nodesx = int(nodesxt/nodest)

        for var_ in vars_:
            if var_ == 'eta':
                # Read water level from NetCDF file
                datax = xr.open_mfdataset(folder + '/case' + str(i).zfill(4) + '/trim-guad.nc', combine='by_coords')
                z = datax.S1.compute().data
                z = z[-1, :, :]  # Use last time step
                data.loc[i, 'eta'] = z[ids_trim]
            else:
                # Read variable from text file at specific grid point
                data.loc[i, var_] = np.loadtxt(folder +'/case' + str(i).zfill(4) + '/' + var_ + '.txt', skiprows=nodesxt - nodesx + 4)[ids[1][0], ids[0][0]]
    
    # Save to compressed CSV file
    save.to_csv(data, filename + str(point[0]) + '_' +  str(point[1]) + '.zip')
    return


def delft_raw_files(folder, vars_, case_id_):
    """Load Delft3D raw output files for a single case.

    Parameters
    ----------
    folder : str or Path
        Directory containing case subdirectories
    vars_ : dict
        Dictionary with variable groups:
        - 'vars_com_guad': Communication module variables
        - 'vars_wavm': Wave module variables
    case_id_ : str
        Case identifier (e.g., 'case0001')

    Returns
    -------
    dict
        Dictionary with variable names as keys and 2D numpy arrays as values

    Notes
    -----
    File format:
    - Text files with headers (first 3 lines + variable-specific header)
    - Line 4 contains: nodesxt, nodesyt, nodest (total nodes in x*t, y*t, t)
    - Data starts at line: nodesxt - nodesx + 5
    - nodesx = nodesxt / nodest
    
    The function processes two variable groups independently, reading
    all files specified in vars_['vars_com_guad'] and vars_['vars_wavm'].

    Examples
    --------
    >>> vars_dict = {
    ...     'vars_com_guad': ['waterlevel', 'velocity_u', 'velocity_v'],
    ...     'vars_wavm': ['hs', 'tp', 'dir']
    ... }
    >>> data = delft_raw_files('./runs', vars_dict, 'case0001')
    >>> print(data['hs'].shape)
    """


    numeric_const_pattern = r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    
    dic = {}
    for var_ in vars_:
        if var_ == 'vars_com_guad':
            fid = open(folder/f"{case_id_}"/f"{vars_['vars_com_guad'][0]}.txt", 'r') 
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt/nodest)
            for j in vars_['vars_com_guad']:
                dic[str(j)] = np.loadtxt(folder/f"{case_id_}"/f"{j}.txt", skiprows=nodesxt - nodesx + 4)
        else:
            fid = open(folder/f"{case_id_}"/f"{vars_['vars_wavm'][0]}.txt", 'r') 
            info = fid.readlines()
            nodesxt, nodesyt, nodest = [int(nodes) for nodes in rx.findall(info[3])]
            nodesx = int(nodesxt/nodest)
            for j in vars_['vars_wavm']:
                dic[str(j)] = np.loadtxt(folder/f"{case_id_}"/f"{j}.txt", skiprows=nodesxt - nodesx + 4)


    return dic


# ---------------------------------------------------------------------------
# DELFT3D-WAVE NEFIS binary extraction (discovered by reverse-engineering)
# ---------------------------------------------------------------------------

# Variable index in the NEFIS .dat binary.
# Each entry: (group, j) where:
#   group 1 → offset = hs_start + j * step
#   group 2 → offset = hs_start + 16*step + j*step   (starts at UBOT)
#   group 3 → offset = hs_start + 28*step + 4 + j*step  (WIND block; 4B = TIME)
_DELFT_VAR_INDEX = {
    # --- Group 1 (map-series, HSIGN onwards) ---
    "hsign":    (1,  0),
    "dir":      (1,  1),
    "pdir":     (1,  2),
    "period":   (1,  3),
    "rtp":      (1,  4),
    "depth":    (1,  5),
    "veloc-y":  (1,  6),
    "veloc-x":  (1,  7),
    "transp-x": (1,  8),
    "transp-y": (1,  9),
    "dspr":     (1, 10),
    "dissip":   (1, 11),
    "leak":     (1, 12),
    "qb":       (1, 13),
    # --- Group 2 (map-series, UBOT onwards; hs_start + 16*step) ---
    "ubot":     (2,  0),
    "steepw":   (2,  1),
    "wlength":  (2,  2),
    "tps":      (2,  3),
    "tm02":     (2,  4),
    "tmm10":    (2,  5),
    # j=6 is DHSIGN (internal, skip)
    "drtm01":   (2,  7),
    "setup":    (2,  8),
    "fx":       (2,  9),
    "fy":       (2, 10),
    # --- Group 3 (WIND block; hs_start + 28*step + 4) ---
    "windu":    (3,  0),
    "windv":    (3,  1),
}

# NEFIS binary layout constants (standard DELFT3D-WAVE output)
_NEFIS_FLOAT_BLOCKS = 30   # 28 map-series + 2 WIND float arrays
_NEFIS_INT_BYTES    = 12   # TIME(4B) + CODE(4B) + TIME_wind(4B)
_NEFIS_FILL         = -9000.0


def _delft_grid_params(case_dir, dat_name, grd_name):
    """Read grid dimensions from .grd and compute hs_start from .dat size."""
    grd_path = case_dir / grd_name
    dat_path = case_dir / dat_name

    mmax = nmax = None
    with open(grd_path, encoding="latin-1") as f:
        for line in f:
            s = line.strip()
            if s.startswith("*") or "=" in s:
                continue
            parts = s.split()
            if len(parts) == 2:
                try:
                    mmax, nmax = int(parts[0]), int(parts[1])
                    break
                except ValueError:
                    continue
    if mmax is None:
        raise ValueError(f"Cannot read grid dimensions from {grd_path}")

    step     = nmax * mmax * 4
    hs_start = dat_path.stat().st_size - _NEFIS_FLOAT_BLOCKS * step - _NEFIS_INT_BYTES + 8
    return {"nmax": nmax, "mmax": mmax, "hs_start": hs_start}


def _delft_var_offset(gp, group, j):
    hs = gp["hs_start"]
    step = gp["nmax"] * gp["mmax"] * 4
    if group == 1:
        return hs + j * step
    if group == 2:
        return hs + 16 * step + j * step
    return hs + 28 * step + 4 + j * step  # group 3: WIND


def _delft_read_var(dat_path, varname, gp, fill_threshold):
    """Read a 2-D float32 array from a NEFIS .dat file."""
    group, j = _DELFT_VAR_INDEX[varname]
    offset   = _delft_var_offset(gp, group, j)
    npts     = gp["nmax"] * gp["mmax"]
    with open(dat_path, "rb") as f:
        f.seek(offset)
        arr = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
    arr[arr < fill_threshold] = np.nan
    return arr.reshape(gp["nmax"], gp["mmax"])


def _delft_read_coords(dat_path, gp, fill_threshold):
    """Read XP, YP coordinates and build active-cell mask."""
    step  = gp["nmax"] * gp["mmax"] * 4
    npts  = gp["nmax"] * gp["mmax"]
    start = gp["hs_start"] + 14 * step   # XP is element 14 from HSIGN
    with open(dat_path, "rb") as f:
        f.seek(start)
        x = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
        y = np.frombuffer(f.read(npts * 4), dtype="<f4").copy().astype(float)
    x = x.reshape(gp["nmax"], gp["mmax"])
    y = y.reshape(gp["nmax"], gp["mmax"])
    active = (x > 0) & (x < 1e6) & (y > 1e6)
    return x, y, active


def extract_delft_wave_dat(
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
        >>> from environmentaltools.processes import extract_delft_wave_dat
        >>>
        >>> # Project points from lat/lon to UTM 30N
        >>> df_pts = pd.read_csv("registro_puntos.csv")
        >>> tr = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
        >>> df_pts["x"], df_pts["y"] = tr.transform(df_pts["longitude"], df_pts["latitude"])
        >>>
        >>> # Load case metadata
        >>> meta = pd.read_csv("500_cases.csv").set_index("id")
        >>>
        >>> df = extract_delft_wave_dat(
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
    from pathlib import Path as _Path
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata

    cases_dir = _Path(cases_dir)

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
        output = _Path(output)
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