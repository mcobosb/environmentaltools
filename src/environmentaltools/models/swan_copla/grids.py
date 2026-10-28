"""Regular grids and result databases (xarray) of the SWAN → COPLA workflow."""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from environmentaltools.spatial.analysis import rotate_coords


def create_db(params, data, mesh="global", time_=None, vars_=None, method="nearest"):
    """Create the project folder with the initialized files for SWAN and Copla models.

    Parameters
    ----------
    params : dict
        Dictionary with the model parameters
    data : pd.DataFrame
        DataFrame with the time series of the boundary data
    mesh : str, optional
        Mesh type identifier. Default is 'global'.
    time_ : array-like, optional
        Time coordinates. Default is None.
    vars_ : str or list, optional
        Variable names to include. Default is None.
    method : str, optional
        Interpolation method ('nearest', 'linear', 'cubic'). Default is 'nearest'.

    Returns
    -------
    xr.Dataset
        Dataset of the project with initialized grids
    """
    lon, lat = create_mesh(params, mesh)
    if isinstance(vars_, str):
        vars_ = [vars_]

    db = create_xarray(lon, lat, time_=time_, vars_=vars_)
    if time_ is None:
        db["depth"][:, :] = griddata(
            data.loc[:, ["x", "y"]], data.loc[:, "z"].values, (lon, lat), method=method
        )
    else:
        db["depth"][:, :, 0] = griddata(
            data.loc[:, ["x", "y"]],
            data.loc[:, "z"].values,
            (lon, lat),
            method="linear",
        )

    return db


def create_mesh(params, mesh="global"):
    """Create a computational mesh for the model domain.

    Parameters
    ----------
    params : dict
        Dictionary containing mesh parameters including:
        
        - {mesh}_length_x : float
            Domain length in x-direction
        - {mesh}_length_y : float
            Domain length in y-direction  
        - {mesh}_nodes_x : int
            Number of nodes in x-direction
        - {mesh}_nodes_y : int
            Number of nodes in y-direction
        - {mesh}_angle : float
            Rotation angle in degrees
        - {mesh}_coords_x : float
            Origin x-coordinate
        - {mesh}_coords_y : float
            Origin y-coordinate
    mesh : str, optional
        Mesh identifier prefix. Default is 'global'.

    Returns
    -------
    lon : np.ndarray
        Longitude coordinates of mesh nodes
    lat : np.ndarray
        Latitude coordinates of mesh nodes
    """

    dx = np.linspace(0, params[mesh + "_length_x"], params[mesh + "_nodes_x"])
    dy = np.linspace(0, params[mesh + "_length_y"], params[mesh + "_nodes_y"])
    dx, dy = np.meshgrid(dx, dy)

    x, y = rotate_coords(dx, dy, params[mesh + "_angle"])
    lon, lat = x + params[mesh + "_coords_x"], y + params[mesh + "_coords_y"]

    return lon, lat


def create_xarray(x, y, time_=None, vars_=None):
    """Create a 2D or 3D xarray Dataset with specified fields.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates (longitude) grid
    y : np.ndarray
        Y-coordinates (latitude) grid
    time_ : array-like or int, optional
        Time coordinates. If int, creates a single time step. Default is None.
    vars_ : str, list, or None, optional
        Variable names to include. Options:
        - None: default variables (depth, Hs, DirM, U, DirU, qc, ql, Setup)
        - 'full': extended variable list including wave parameters
        - list: custom list of variable names
        Default is None.

    Returns
    -------
    xr.Dataset
        Dataset with initialized variables and coordinates
    """
    dict_ = {}

    if vars_ is None:
        vars_ = "depth", "Hs", "DirM", "U", "DirU", "qc", "ql", "Setup"
    elif vars_ == "full":
        vars_ = (
            "depth",
            "Sb",
            "Hs",
            "Tp",
            "DirM",
            "Setup",
            "cp",
            "L",
            "U",
            "DirU",
            "Dr",
            "Db",
            "Df",
            "Qst",
            "Qbt",
            "qc",
            "ql",
        )

    if time_ is not None:
        if isinstance(time_, int):
            zeros = np.zeros(np.append(np.shape(x), 1))
            time_ = [time_]
        else:
            zeros = np.zeros(np.append(np.shape(x), len(time_)))
        for i in vars_:
            dict_[i] = (["lon", "lat", "time"], zeros.copy())
        db = xr.Dataset(
            dict_,
            coords={"x": (["lon", "lat"], x), "y": (["lon", "lat"], y), "time": time_},
        )
    else:
        zeros = np.zeros(np.shape(x))
        for i in vars_:
            dict_[i] = (["lon", "lat"], zeros.copy())
        db = xr.Dataset(
            dict_, coords={"x": (["lon", "lat"], x), "y": (["lon", "lat"], y)}
        )

    return db


def save_db(time_, grid, data, params):
    """Create and save xarray Dataset with computed model results for a time step.

    Updates the database by creating an xarray Dataset with the computation results
    and saves it to a NetCDF file.

    Parameters
    ----------
    time_ : int or pd.DatetimeIndex
        Time step identifier
    grid : dict
        Dictionary containing computed values for the current time step including:
        
        - depth : np.ndarray
            Water depth
        - Hs : np.ndarray
            Significant wave height
        - DirM : np.ndarray
            Mean wave direction
        - U : np.ndarray
            Current velocity magnitude
        - DirU : np.ndarray
            Current direction
        - qc, ql : np.ndarray
            Cross-shore and longshore sediment transport
        - Setup : np.ndarray
            Wave setup
        - Qb : np.ndarray
            Bed load transport
    data : pd.DataFrame
        DataFrame with boundary data
    params : dict
        Dictionary with model parameters including:
        
        - fileout : str
            Output file path prefix

    Returns
    -------
    xr.Dataset
        Updated dataset with computed fields
    """
    # kw = ['Hs', 'qc', 'ql', 'Setup', 'DirM', 'd', 'cp',
    #     'Dr', 'Db', 'Df', 'L', 'U', 'DirU', 'Qst', 'Qbt']
    kw = ["depth", "Hs", "DirM", "U", "DirU", "qc", "ql", "Setup", "Qb"]

    db = create_db(params, data, "local", vars_=kw, time_=time_)

    for var_ in kw:
        if var_.startswith("Dir"):
            db[var_][:, :, 0] = np.remainder(270 - grid[var_], 360)
        else:
            db[var_][:, :, 0] = grid[var_]

    db.to_netcdf(params["fileout"] + "_" + str(time_).zfill(4) + ".nc")

    return db
