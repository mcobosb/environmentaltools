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