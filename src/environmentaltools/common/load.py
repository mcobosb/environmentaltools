import re

import numpy as np
import pandas as pd
import xarray as xr
from environmentaltools.common import read, save
from scipy.io import loadmat as ldm


def create_mesh_dictionary(file_name: str, sheet_name: str = None):
    """Read Excel file and create mesh parameter dictionary.

    Loads mesh configuration from an Excel file and optionally extracts
    a specific sheet as a dictionary.

    Args:

        file_name (str): Path to the Excel file with mesh parameters.
        sheet_name (str, optional): Name of specific sheet to extract as dictionary.
            If None, returns the entire DataFrame. Defaults to None.

    Returns:
        dict or pd.DataFrame: Dictionary of parameters if sheet_name specified,
            otherwise returns full DataFrame.
    """
    info = read.xlsx(file_name)
    if sheet_name is not None:
        params = info[sheet_name].to_dict()
    else:
        params = info

    return params


def cshore_config():
    """Create default CSHORE model configuration parameters.

    Generates a dictionary with default configuration parameters for the CSHORE
    (Cross-shore) numerical model, including morphology, wave, and sediment
    transport settings.

    Returns:
        dict: Dictionary containing CSHORE configuration parameters including:
            - Model flags (iline, iprofl, isedav, etc.)
            - Physical parameters (gamma, sporo, sg, etc.)
            - Boundary conditions (timebc_wave, swlbc, etc.)
            - Sediment properties (tanphi, blp, slp, etc.)
    """
    # Common configuration file
    props = {}
    props["iline"] = 1  # 1 = single line
    props["iprofl"] = (
        1  # 0 = no morph, 1 = run morph, 1.1 = run morph without initial smoothing
    )
    props["isedav"] = 0  # 0 = unlimited sand, 1 = hard bottom
    props["iperm"] = 0  # 0 = no permeability, 1 = permeable
    props["iover"] = 1  # 0 = no overtopping , 1 = include overtopping
    props["infilt"] = 0  # 1 = include infiltration landward of dune crest
    props["iwtran"] = (
        0  # 0 = no standing water landward of crest, 1 = wave transmission due to overtopping
    )
    props["ipond"] = 0  # 0 = no ponding seaward of SWL
    props["iwcint"] = 1  # 0 = no W & C interaction , 1 = include W & C interaction
    props["iroll"] = 1  # 0 = no roller, 1 = roller
    props["iwind"] = 0  # 0 = no wind effect
    props["itide"] = 0  # 0 = no tidal effect on currents
    props["iclay"] = 0  # Clay layer option
    props["iveg"] = 0  # Vegetation effect (0 = no vegetation, 1 = include vegetation)
    props["veg_Cd"] = 1  # Vegetation drag coefficient
    props["veg_n"] = 100  # Vegetation density (stems per m²)
    props["veg_dia"] = 0.01  # Vegetation stem diameter (m)
    props["veg_ht"] = 0.20  # Vegetation height (m)
    props["veg_rod"] = 0.1  # Vegetation erosion limit below sand for failure (m)
    props["veg_extent"] = [
        0.7,
        1,
    ]  # Vegetation coverage as fraction of total domain length
    props["gamma"] = 0.8  # shallow water ratio of wave height to water depth
    props["sporo"] = 0.4  # sediment porosity
    props["sg"] = 2.65  # specific gravity
    props["effb"] = 0.005  # suspension efficiency due to breaking eB
    props["efff"] = 0.01  # Suspension efficiency due to friction (ef)
    props["slp"] = 0.5  # Suspended load parameter
    props["slpot"] = 0.1  # Overtopping suspended load parameter
    props["tanphi"] = 0.630  # Tangent of sediment friction angle (degrees)
    props["blp"] = 0.001  # Bedload transport parameter
    props["rwh"] = 0.02  # Numerical runup wire height (m)
    props["ilab"] = 1  # Controls boundary condition timing (don't change)
    props["fric_fac"] = 0.015  # Bottom friction factor

    # Boundary conditions and timing parameters
    props["timebc_wave"] = 3600  # Wave boundary condition time interval (seconds)

    props["timebc_surg"] = props["timebc_wave"]  # Surge boundary condition time
    props["nwave"] = 1  # Number of wave time steps
    props["nsurg"] = props["nwave"]  # Number of surge time steps

    props["Wsetup"] = 0  # Wave setup at seaward boundary (meters)
    props["swlbc"] = 0.0  # Water level at seaward boundary (meters)

    return props


def read_cshore(file_type: str, path: str, skiprows: int = 1):
    """Read CSHORE model output files.

    Loads and parses output files from the CSHORE numerical model, supporting
    various output types (profiles, transport rates, energy, velocities, etc.).

    Args:
        file_type (str): Type of CSHORE output file to read. Options include:
            'bprof', 'bsusl', 'cross', 'energ', 'longs', 'param', 'rolle',
            'setup', 'swase', 'timse', 'xmome', 'xvelo', 'ymome', 'yvelo'.
        path (str): Directory path containing CSHORE output files.
        skiprows (int): Number of rows to skip when reading file. Defaults to 1.

    Returns:
        pd.DataFrame: Parsed CSHORE output data with appropriate column names
            and spatial index (x-distance in meters).
    """
    header = {
        "bprof": ["z"],
        "bsusl": [r"$P_b$", r"$P_s$", r"$V_s$"],
        "cross": [r"$Q_{b,x}$", r"$Q_{s,x}$", r"$Q_{b,x} + Q_{s,x}$"],
        "crvol": [],
        "energ": [r"Eflux (m3/s)", "Db (m2/s)", "Df (m2/s)"],
        "longs": [r"$Q_{b,y}$", r"$Q_{s,y}$", r"$Q_{b,y} + Q_{s,y}$"],
        "lovol": [],
        "param": ["T (s)", r"$Q_b$ (nondim)", "Sigma* (nondim)"],
        "rolle": ["Rq (m2/s)"],
        "setup": [r"$\eta + S_{tide}$ (m)", "d (m)", r"$\sigma_{eta}$ (m)"],
        "swase": ["de (m)", "Uxe (m/s)", "Qxe (m2/s)"],
        "timse": ["t (id)", "t (s)", "q0 (m2/s)", "qbx,lw (m2/s)", "qsx,lw (m2/s)"],
        "xmome": ["Sxx (m2)", "taubx (m)"],
        "xvelo": [r"$U_x$", r"$U_{x,std}$"],
        "ymome": ["Sxx (m2)", "taubx (m)"],
        "yvelo": ["sin theta (unitary)", r"$U_y$", r"$U_{y,std}$"],
    }

    # TODO: include morphology options
    # EWD: Output exceedance probability 0.015
    # q0: wave overtopping rate, qbx,lw: cross-shore bedload transport rate at the landward end of the computation domain
    
    # Construct output filename (CSHORE prepends 'O' to output files)
    filename = path + "/" + "O" + file_type.upper()
    
    if file_type == "bprof":
        # For beach profile, read number of valid points from header
        fid = open(filename, "rb")
        properties = fid.readline()
        id_ = int(properties.split()[1])
        df = pd.read_csv(
            filename,
            sep="\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_type],
        )
        # Trim to actual number of profile points
        df = df.iloc[:id_, :]
    else:
        # Standard reading for other output types
        df = pd.read_csv(
            filename,
            sep="\s+",
            skiprows=skiprows,
            index_col=0,
            names=header[file_type],
        )
    # Index represents x distance in meters

    # Ensure column names are strings
    df.columns = df.columns.astype("str")

    return df


def read_copla(file_name: str, grid: dict = None):
    """Read COPLA model velocity output files.

    Loads velocity field data from COPLA
    model output and computes magnitude and direction on a 2D grid with ghost cells.

    Args:
        file_name (str): Path to COPLA velocity output file.
        grid (dict, optional): Existing grid dictionary to update. If None,
            creates new dictionary. Defaults to None.

    Returns:
        dict: Dictionary containing velocity components and derived fields:
            - 'u': East-west velocity component (m/s) with ghost cells
            - 'v': North-south velocity component (m/s) with ghost cells
            - 'U': Velocity magnitude (m/s)
            - 'DirU': Velocity direction (degrees, nautical convention)
    """
    data = pd.read_csv(
        file_name,
        skiprows=7,
        delim_whitespace=True,
        header=None,
        index_col=0,
        names=["x", "y", "u", "v"],
    )
    # Create meshgrid for spatial dimensions
    _, x = np.meshgrid(data.y.unique(), data.x.unique())

    if grid is None:
        grid = {}

    grid = dict()
    nx, ny = np.shape(x)
    
    # Create velocity arrays with ghost cells (boundary padding)
    for var_ in ["u", "v"]:
        grid[var_] = np.zeros([nx + 2, ny + 2])
        # Fill interior cells with data
        grid[var_][1:-1, 1:-1] = data[var_].to_numpy().reshape([nx, ny])

    # Compute velocity magnitude
    grid["U"] = np.sqrt(grid["u"] ** 2 + grid["v"] ** 2)
    
    # Compute velocity direction (nautical convention: 0° = North, clockwise)
    grid["DirU"] = np.fmod(np.rad2deg(np.arctan2(grid["v"], grid["u"])) + 90, 360)

    return grid


def read_swan(file_name: str, grid: dict = None, variables: list = None):
    """Read SWAN model output from MATLAB file format.

    Loads wave field data from SWAN (Simulating WAves Nearshore) model output
    stored in MATLAB format and computes wave number from wavelength.

    Args:
        file_name (str): Path to MATLAB (.mat) file containing SWAN output.
        grid (dict, optional): Existing grid dictionary to update. If None,
            creates new dictionary. Defaults to None.
        variables (list, optional): List of output variable names in order:
            [x, y, depth, Qb, L, Setup, Hs, DirM]. If None, uses default names.
            Defaults to None.

    Returns:
        dict: Dictionary containing wave parameters on grid:
            - 'x', 'y': Coordinates
            - 'depth': Water depth (m)
            - 'Qb': Wave breaking dissipation
            - 'L': Wavelength (m)
            - 'Setup': Wave setup (m)
            - 'Hs': Significant wave height (m)
            - 'DirM': Mean wave direction (degrees)
            - 'kp': Wave number (2π/L) computed from wavelength
    """
    if not variables:
        variables = ["x", "y", "depth", "Qb", "L", "Setup", "Hs", "DirM"]

    if grid is None:
        grid = {}

    # Load MATLAB file with SWAN output
    swan_dictionary = ldm(file_name)
    
    # Map SWAN variable names to user-specified names
    for ind_, var_ in enumerate(
        ["Xp", "Yp", "Depth", "Qb", "Wlen", "Setup", "Hsig", "Dir"]
    ):
        grid[variables[ind_]] = swan_dictionary[var_]
        # Replace NaN values with small number to avoid numerical issues
        grid[variables[ind_]][np.isnan(grid[variables[ind_]])] = 1e-6

    # Compute wave number from wavelength
    grid["kp"] = 2 * np.pi / grid["L"]

    return grid


def delft_raw_files_point(
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
        filename (str): Output filename prefix. Defaults to "seastates".

    Returns:
        None: Saves extracted data to ZIP file with format:
            {filename}{point[0]}_{point[1]}.zip
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
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
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
    save.to_csv(data, filename + "_" + str(point[0]) + "_" + str(point[1]) + ".zip")
    return


def delft_raw_files(folder: str, variables: dict, case_id: str):
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
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
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