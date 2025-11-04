import os
import shutil
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr
from loguru import logger
from environmentaltools.processes import load, waves, write
from environmentaltools.spatial.analysis import rotate_coords
from environmentaltools.common import utils, read, save
from scipy.interpolate import griddata, interp1d
from scipy.special import erfc


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


def slopes(data, variable="DirU"):
    """Compute the bottom slopes in the direction of a specified variable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing geometry data with columns 'x', 'y', 'depth', and the specified variable
    variable : str, optional
        Name of the circular direction variable in degrees. Default is 'DirU'.

    Returns
    -------
    np.ndarray
        Array of bottom slopes in the specified direction
    """
    xi, yi = (
        (data["x"] + np.cos(data[variable] * np.pi / 180)).reshape(-1),
        (data["y"] + np.sin(data[variable] * np.pi / 180)).reshape(-1),
    )
    Sb = griddata(
        (data["x"].reshape(-1), data["y"].reshape(-1)),
        data["depth"].reshape(-1),
        (xi, yi),
    ) - data["depth"].reshape(-1)
    Sb = Sb.reshape(np.shape(data["x"]))
    Sb[np.isnan(Sb)] = 0
    return Sb


def sediment_transport_Kobayashi(j, data, grid, params):
    """Calculate suspended and bed-load transport following Kobayashi et al. (2008).

    This function implements the Kobayashi sediment transport model which computes
    both suspended load and bed load transport rates based on wave and current conditions.

    Parameters
    ----------
    j : pd.DatetimeIndex
        Timestamp for the current computation
    data : pd.DataFrame
        DataFrame with input time series including wave parameters
    grid : dict
        Dictionary containing 2D numpy arrays with required spatial data (depth, Setup, etc.)
    params : dict
        Dictionary with model constants and parameters including:
        
        - fb : float
            Bottom friction coefficient
        - wf : float
            Fall velocity
        - gamma : float
            Breaking parameter
        - And other model-specific parameters

    Returns
    -------
    dict
        Updated grid dictionary with computed sediment transport fields including:
        - Qst : Suspended load transport rate
        - Qbt : Bed load transport rate
        - qc : Cross-shore sediment transport
        - ql : Alongshore sediment transport
    """

    G = 9.8091
    grid["D"] = grid["depth"] + grid["Setup"]
    grid["D"][grid["D"] < 0.1] = np.nan  # Total depth
    grid["cp"] = np.sqrt(G * grid["D"])  # Shallow water waves
    grid["sigma"] = data.loc[j, "Hs"] / np.sqrt(8) / grid["D"] * grid["cp"]
    grid["sigma"][np.isnan(grid["sigma"])] = 0

    grid["Rs"] = (2 / params["fb"]) ** (1 / 3) * params["wf"] / grid["sigma"]
    grid["Rs"][np.isnan(grid["Rs"])] = 0
    grid["Ps"] = 0.5 * erfc((grid["Rs"] + grid["U"]) / np.sqrt(2)) + 0.5 * erfc(
        (grid["Rs"] - grid["U"]) / np.sqrt(2)
    )

    grid["Hb"] = (
        0.88
        / grid["kp"]
        * np.tanh(
            params["gamma"]
            * grid["kp"]
            * grid["D"]
            / (0.88 * grid["cp"] * data.loc[j, "Tp"])
        )
    )  # Kobayashi 0.88
    grid["Hb"][grid["Hb"] / grid["D"] > 0.88] = (
        0.88 * grid["D"][grid["Hb"] / grid["D"] > 0.88]
    )
    grid["Hb"][np.isnan(grid["Hb"])] = 0

    #   Suspended load transport
    # --------------------------------------------------------------------------
    grid["Dr"] = (
        G
        * (params["rho"] * 0.9 * grid["Hb"] ** 2 * grid["cp"] / data.loc[j, "Tp"])
        * np.sin(params["br"])
    )  # Roller effect
    grid["Dr"][np.isnan(grid["Dr"])] = 0

    grid["arot"] = 1 / 3 * grid["Sb"] * data.loc[j, "Tp"] * np.sqrt(G / grid["D"])
    grid["arot"][np.isnan(grid["arot"])] = 0
    grid["Db"] = (
        params["rho"]
        * grid["arot"]
        * G
        * grid["Qb"]
        * grid["Hb"] ** 2
        / (4 * data.loc[j, "Tp"])
    )  # Battjes and Stive, 1985
    grid["Df"] = 0.5 * params["rho"] * G * params["fb"] * grid["U"] ** 3
    grid["Vs"] = (
        (params["eb"] * (grid["Db"] - grid["Dr"]) + params["ef"] * grid["Df"])
        * grid["Ps"]
        / (params["rho"] * G * (params["S"] - 1) * params["wf"])
    )
    grid["Qst"] = params["aK"] * grid["U"] * grid["Vs"] * np.sqrt(1 + grid["Sb"] ** 2)

    #    Bed-load transport
    # --------------------------------------------------------------------------
    grid["Rb"] = (
        np.sqrt(
            2 * G * (params["S"] - 1) * params["d50"] * params["Shic"] / params["fb"]
        )
        / grid["sigma"]
    )
    grid["Rb"][np.isnan(grid["Rb"])] = 0
    grid["Pb"] = 0.5 * erfc((grid["Rb"] + grid["U"]) / np.sqrt(2)) + 0.5 * erfc(
        (grid["Rb"] - grid["U"]) / np.sqrt(2)
    )

    grid["Gs"] = (np.tan(params["phi"]) - 2 * grid["Sb"]) / (
        np.tan(params["phi"]) - grid["Sb"]
    )
    grid["Gs"][grid["Sb"] < 0] = np.tan(params["phi"]) / (
        np.tan(params["phi"]) + grid["Sb"][grid["Sb"] < 0]
    )
    grid["Qbt"] = (
        params["bK"]
        * grid["Pb"]
        * grid["Gs"]
        * grid["sigma"] ** 3
        / (G * (params["S"] - 1))
    )  # Formulación de Meyer-Peter-Mueller, modificado

    # Cross-shore and alongshore sediment transport
    # --------------------------------------------------------------------------
    grid["qc"] = (
        (grid["Qbt"] + grid["Qst"])
        * np.sin((grid["DirU"] - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][0])
    grid["ql"] = (
        (grid["Qbt"] + grid["Qst"])
        * np.cos((grid["DirU"] - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][1])

    return grid


def sediment_transport_CERC(data, params, theta_c):
    """Calculate longshore sediment transport using CERC formula.

    The CERC (Coastal Engineering Research Center) formula is used to estimate
    longshore sediment transport rates based on breaking wave characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with wave parameters including:
        
        - Hr : Breaking wave height (m)
        - thetar : Breaking wave angle (degrees)
        
    params : dict
        Dictionary with sediment and water properties:
        
        - d50 : float
            Mean sediment grain size (m)
        - rho : float
            Fresh-water density (kg/m³)
        - rho_s : float
            Sediment density (kg/m³)
        - n : float
            Porosity
        - gamma : float
            Breaking parameter
            
    theta_c : float
        Beach angle (degrees)

    Returns
    -------
    sign_ : np.ndarray
        Sign of transport direction
    Q : np.ndarray
        Longshore sediment transport rate
    """

    data["alphar"] = np.deg2rad(
        np.min(
            np.asarray(
                [np.abs(data["thetar"] - theta_c), np.abs(theta_c - data["thetar"])]
            ),
            axis=0,
        )
    )
    data.loc[data["alphar"] > np.pi / 2, "alphar"] = (
        0  # el valor absolutio del ángulo entre la normal y el tren de olas no puede ser superior a 90
    )

    G = 9.8091
    K = 1.4 * np.exp(-2.5 * params["d50"])
    Q0 = (
        K
        * (
            params["rho"]
            * np.sqrt(G)
            / (
                16
                * np.sqrt(params["gamma"])
                * (params["rho_s"] - params["rho"])
                * (1 - params["n"])
            )
        )
        * data["Hr"] ** 2.5
    )
    sign_ = np.sign(data["thetar"] - theta_c)
    Q = sign_ * Q0 * np.sin(2 * data["alphar"])
    return sign_, Q


def nesting(k, l, params, data):
    """Execute nested SWAN and COPLA model runs for a specific time step.

    This function performs a coupled wave-morphodynamic simulation by first running
    SWAN for wave propagation, then COPLA for profile evolution, and finally computing
    sediment transport using the Kobayashi method.

    Parameters
    ----------
    k : int
        Time step index
    l : pd.DatetimeIndex or similar
        Current timestamp for the computation
    params : dict
        Dictionary with model parameters including:
        
        - cwd : str
            Current working directory
        - directory : str
            Project directory name
        - Other model-specific parameters
        
    data : pd.DataFrame
        DataFrame with boundary condition time series

    Returns
    -------
    dict
        Grid dictionary containing computed fields including wave parameters,
        velocities, and sediment transport rates
    """
    id_ = str(k + 1).zfill(4)
    write.write_swan(k, l, id_, data, params, mesh="local")
    current_working_directory = os.path.join(params["cwd"], params["directory"], id_)

    if not os.path.exists(os.path.join(current_working_directory, id_ + ".mat")):
        run_swan(current_working_directory)

    write.copla(k, l, str(k + 1).zfill(4), data, params, mesh="local")
    copla(current_working_directory)
    # join(current_working_directory)

    # grid = load.copla(params['directory'] + '/' + id_ + '/' + id_+ 'tot.out')
    grid = load.read_copla(params["directory"] + "/" + id_ + "/" + id_ + "vel.001")
    grid = load.read_swan(params["directory"] + "/" + id_ + "/" + id_ + ".mat", grid)

    grid["Sb"] = slopes(grid)  # compute the slopes in the wave current direction
    grid = sediment_transport_Kobayashi(
        l, data, grid, params
    )  # compute the sediment transport

    return grid


def run_swan(swan_executable_path, working_directory):
    """Execute SWAN (Simulating WAves Nearshore) model.

    Runs the SWAN wave model in the specified working directory. SWAN is a third-generation
    wave model for obtaining realistic estimates of wave parameters in coastal areas.

    Parameters
    ----------
    swan_executable_path : str
        Path to the SWAN executable file
    working_directory : str
        Path to the working directory containing SWAN input files

    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If SWAN executable is not found at the specified path
    """
    # Verify that SWAN executable exists
    if not os.path.isfile(swan_executable_path):
        raise FileNotFoundError(
            f"SWAN executable not found at: {swan_executable_path}\n"
            "Please download SWAN from: https://swanmodel.sourceforge.io/download/download.htm\n"
            "After installation, provide the correct path to the SWAN executable."
        )
    
    subprocess.run(swan_executable_path, cwd=working_directory)

    return


def run_copla(copla_executable_path, working_directory):
    """Execute COPLA (Coastal Profile Algorithm) model.

    Runs the COPLA morphodynamic model which simulates cross-shore profile evolution
    in the specified working directory.

    Parameters
    ----------
    copla_executable_path : str
        Path to the COPLA executable file
    working_directory : str
        Path to the working directory containing COPLA input files

    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If COPLA executable is not found at the specified path
    """
    # Verify that COPLA executable exists
    if not os.path.isfile(copla_executable_path):
        raise FileNotFoundError(
            f"COPLA executable not found at: {copla_executable_path}\n"
            "Please ensure COPLA is properly installed and provide the correct path to the executable."
        )
    
    subprocess.run(copla_executable_path, cwd=working_directory)
    return


def run_cshore(cshore_executable_path, working_directory):
    """Execute CSHORE (Coastal Storm Hindcast of Reshaping and Erosion) model.

    Runs the CSHORE model for simulating beach profile evolution and storm impacts
    in the specified working directory.

    Parameters
    ----------
    cshore_executable_path : str
        Path to the CSHORE executable file
    working_directory : str
        Path to the working directory containing CSHORE input files

    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If CSHORE executable is not found at the specified path
    """
    # Verify that CSHORE executable exists
    if not os.path.isfile(cshore_executable_path):
        raise FileNotFoundError(
            f"CSHORE executable not found at: {cshore_executable_path}\n"
            "Please download CSHORE from: https://github.com/erdc/cshore\n"
            "After installation, provide the correct path to the CSHORE executable."
        )
    
    subprocess.run(cshore_executable_path, cwd=working_directory)

    return


def run_coastalme(coastalme_executable_path, working_directory):
    """Run CoastalME model in the specified directory.

    Parameters
    ----------
    coastalme_executable_path : str
        Path to the CoastalME executable file
    working_directory : str
        Path to the working directory where CoastalME will be executed
    
    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        If CoastalME executable is not found at the specified path
    """
    # Verify that CoastalME executable exists
    if not os.path.isfile(coastalme_executable_path):
        raise FileNotFoundError(
            f"CoastalME executable not found at: {coastalme_executable_path}\n"
            "Please download CoastalME from: https://github.com/coastalme/\n"
            "After installation, provide the correct path to the CoastalME executable."
        )

    subprocess.run(coastalme_executable_path, cwd=working_directory)
    return


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


def clean(params):
    """Remove temporary computation directory tree if deletion is enabled.

    Cleans up the working directory created during model execution by recursively
    removing all files and subdirectories.

    Parameters
    ----------
    params : dict
        Dictionary with model parameters including:
        
        - delete_folder : bool
            Flag to enable/disable directory deletion
        - cwd : str
            Current working directory
        - directory : str
            Name of the directory to remove

    Returns
    -------
    None
    """
    if params["delete_folder"]:
        shutil.rmtree(
            os.path.join(params["cwd"], params["directory"]), ignore_errors=True
        )
    return


def equilibrium_plan_shape(params, data):
    """Compute equilibrium planform shape using parabolic bay equation.

    Calculates the equilibrium beach planform shape based on wave diffraction
    theory and parabolic bay equation (Hsu and Evans, 1989). Useful for
    assessing coastal stability and headland bay beach morphology.

    Parameters
    ----------
    params : dict
        Dictionary with parameters including:
        
        - x : float
            X-coordinate of diffraction point
        - y : float
            Y-coordinate of diffraction point
        - theta_m : float
            Mean wave direction (degrees)
        - Ts12 : float
            Mean wave period (s)
        - h : float
            Water depth (m)
        - beta_r : float, optional
            Parabolic coefficient (default: 2.13)
            
    data : pd.DataFrame
        DataFrame with profile coordinates containing:
        
        - x : X-coordinates
        - y : Y-coordinates

    Returns
    -------
    x : np.ndarray
        X-coordinates of equilibrium planform
    y : np.ndarray
        Y-coordinates of equilibrium planform
    theta_0 : float
        Control angle at the reference point (radians)
    """

    if not "beta_r" in params.keys():
        params["beta_r"] = 2.13

    theta_m = np.remainder(270 - params["theta_m"], 360)

    # compute the angles between the diffraction point and every point along the profile
    thetas = np.arctan2(data.x - params["x"], data.y - params["y"])
    theta_0 = thetas - np.deg2rad(theta_m - 90)
    ds = np.sqrt((data.x - params["x"]) ** 2 + (data.y - params["y"]) ** 2)
    Ys = ds * np.cos(theta_0)

    npoint = np.argmin(
        np.abs(thetas)
    )  # plane that follows the mean energy flux and crosses the diffraction point

    k = waves.wave_number(params["Ts12"], params["h"])
    L = 2 * np.pi / k

    diff, npoints, iter_ = 1, list(), 0
    while diff > 1e-3:
        alpha_min = np.arctan(
            np.sqrt(
                params["beta_r"] ** 4 / 16
                + params["beta_r"] ** 2 * Ys[npoint] / (2 * L)
            )
            / (Ys[npoint] / L)
        )
        npoint = np.argmin(np.abs(alpha_min - theta_0))
        npoints.append(npoint)

        diff = np.abs(alpha_min - theta_0[npoint])
        if any(npoints == npoint):
            break

        iter_ += 1

    beta = np.pi / 2 - theta_0[npoint]
    R0 = ds[npoint]
    R0 = 254.8474172725602  # change this value for convex beaches
    coeffs = read.xlsx("parabolic_coeffs")
    C0 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C0"])
    C1 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C1"])
    C2 = np.interp(np.rad2deg(beta), coeffs.index, coeffs["C2"])

    theta = np.deg2rad(np.linspace(np.rad2deg(beta), 180, 100))
    R = R0 * (C0 + C1 * (beta / theta) + C2 * (beta / theta) ** 2)

    theta = np.pi / 2 - theta + np.deg2rad(theta_m - 90)
    x = -R * np.sin(theta)
    y = R * np.cos(theta)

    # print(L, np.rad2deg(theta_0[npoint]))
    return x + params["x"], y + params["y"], theta_0[npoint]


def coastline_evolution(coastlines, points):
    """Track coastline position changes over time at specific points.

    Computes the distance of coastline displacement at multiple monitoring points
    throughout a time series, measuring retreat or advance perpendicular to the
    initial coastline position.

    Parameters
    ----------
    coastlines : dict
        Dictionary with time steps as keys and DataFrames with coastline coordinates
        as values. Each DataFrame should contain 'x' and 'y' columns.
    points : list
        List of 2D points (x, y tuples or arrays) where coastline evolution is tracked

    Returns
    -------
    pd.DataFrame
        DataFrame with coastline evolution distances at each monitoring point.
        Positive values indicate advance, negative values indicate retreat.
    """

    no_points = len(points)
    no_steps = len(coastlines)
    evolution = pd.DataFrame(-1, index=range(1, no_steps), columns=range(no_points))

    for ind_, point in enumerate(points):
        index_0 = utils.nearest(coastlines[0], point)
        for time_ in coastlines.keys():
            index_ = utils.nearest(
                coastlines[time_],
                [coastlines[0].loc[index_0, "x"], coastlines[0].loc[index_0, "y"]],
            )
            if (
                coastlines[time_].loc[index_, "y"] - coastlines[0].loc[index_0, "y"]
            ) > 0:
                constant = -1
            else:
                constant = 1
            evolution.loc[time_, ind_] = constant * np.sqrt(
                (coastlines[time_].loc[index_, "x"] - coastlines[0].loc[index_0, "x"])
                ** 2
                + (coastlines[time_].loc[index_, "y"] - coastlines[0].loc[index_0, "y"])
                ** 2
            )

    return evolution


def precipitation_to_flow(data, info):
    """Convert precipitation time series to streamflow using SCS Curve Number method.

    Transforms rainfall data into runoff hydrographs using the NRCS (formerly SCS)
    Curve Number method with unit hydrograph theory. Accounts for soil moisture
    conditions, infiltration, and routing.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Precipitation time series with datetime index
    info : dict
        Dictionary with watershed and model parameters:
        
        - lon : float
            Main river length (km)
        - slope : float
            Main river slope (m/m)
        - area : float
            Watershed area (km²)
        - cn2 : float
            SCS Curve Number for average moisture conditions (0-100)
        - k_ac : float, optional
            Aquifer storage constant (hours). Default: 360
        - f_abs : float, optional
            Absorption factor. Default: 1
        - rainning_pattern : str, optional
            SCS precipitation pattern ID. Default: '1_24h'
        - events : bool, optional
            Whether data contains discrete events. Default: False
        - freq_raw_data : str, optional
            Frequency of raw data ('D', 'H', etc.). Default: 'D'
        - dt : float, optional
            Time step (hours). Default: 1
        - model : str
            Concentration time model: 'SCS', 'Temez', 'Kirpich',
            'Kirpich-natural_slope'

    Returns
    -------
    pd.DataFrame
        DataFrame with computed hydrological variables including:
        - pr : Distributed precipitation
        - cumulative : Cumulative precipitation by events
        - cn : Dynamic curve number
        - net_pr : Effective precipitation (runoff)
        - sup_flow : Surface flow (m³/s)
        - base_flow : Base flow (m³/s)
        - total_flow : Total streamflow (m³/s)
        - infiltration : Infiltrated water depth
        - Mass balance statistics

    Raises
    ------
    ValueError
        If required parameters are missing or if concentration time model is not recognized
    """

    if ((not "lon") | (not "slope") | (not "area") | (not "cn2")) in info.keys():
        raise ValueError(
            "Watershed area and mean curve number, main river length and slope are required"
        )

    if not "k_ac" in info.keys():
        info["k_ac"] = 360

    if not "f_abs" in info.keys():
        info["f_abs"] = 1

    if not "rainning_pattern" in info.keys():
        info["rainning_pattern"] = "1_24h"
        logger.info(
            "The rainning pattern was not defined. Assuming a 24-hours evenly distributed pattern."
        )

    if not "events" in info.keys():
        info["events"] = False

    if not "freq_raw_data" in info.keys():
        info["freq_raw_data"] = "D"

    if not "dt" in info.keys():
        info["dt"] = 1

    if not "model" in info.keys():
        raise ValueError(
            "Model of evaluation of concretation times is required. Options are SCS, Temez, Kirpich, and Kirpich-natural_slope"
        )

    info["cn3"] = 23 * info["cn2"] / (10 + 0.13 * info["cn2"])
    info["cn1"] = 4.2 * info["cn2"] / (10 - 0.058 * info["cn2"])

    data = data * info["f_abs"]

    pattern_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "src",
        "patrones_SCS",
    )

    # Read one of the SCS precipitation patterns
    pattern = read.xlsx(pattern_path)[info["rainning_pattern"]].values

    pattern[1:] = pattern[1:] - pattern[:-1]

    # Create the dataframe for the whole period
    ini = datetime(
        data.index[0].year, data.index[0].month, data.index[0].day, data.index[0].hour
    )
    end = datetime(data.index[-1].year, data.index[-1].month, data.index[-1].day, 23)
    index = pd.date_range(ini, end, freq="H")
    df = pd.DataFrame(0, index=index, columns=["pr"])

    # Distribute the precipitation along the day
    df = distribute_precipitation(data, pattern, df, info)

    # Compute the cumulative sum by events
    df["cumulative"] = cumulative_by_events(df)

    window_size = 5 + 1  # 5 hours
    humedo = (
        df["cumulative"]
        .rolling(window_size)
        .apply(wet_soil, args=(info["cn3"],), raw=True)
    )

    window_size = (5 * 24) + 1  # 5 days into hours
    seco = (
        df["cumulative"]
        .rolling(window_size)
        .apply(dry_soil, args=(info["cn1"],), raw=True)
    )

    df["cn"] = 0
    df["cn"] = humedo.fillna(info["cn2"])
    df["cn"].loc[seco.notnull()] = seco

    df["f_a"] = 254.0 * (100 / df["cn"] - 1)  # Fraction of cumulative abstraction
    df["i_a"] = 0.2 * df["f_a"]  # Abstraccion Inicial --> A partir de SCS

    df["effec_cum"] = 0
    df["effec_cum"].loc[df["cumulative"] >= df["i_a"]] = (
        df["cumulative"].loc[df["cumulative"] >= df["i_a"]]
        - df["i_a"].loc[df["cumulative"] >= df["i_a"]]
    ) ** 2 / (df["cumulative"] + 0.8 * df["f_a"].loc[df["cumulative"] >= df["i_a"]])

    df["net_pr"] = 0
    diff = np.diff(df["effec_cum"])
    df["net_pr"].iloc[1:] = np.where(diff > 0, diff, 0)

    df = unit_hydrograph_model(df, info)

    # BASE FLOW
    # Computing the infiltrated flow
    df["infiltration"] = df["pr"] - df["net_pr"]
    df["infil_flow"] = (
        df["infiltration"] * info["area"] * 1e6 / (1000 * 3600)
    )  # mm/h --> m3/s

    df["base_flow"] = 0
    df["base_flow"].acumulado = df["base_flow"].iloc[0]

    window_size = 2
    df["base_flow"] = (
        df["base_flow"]
        .rolling(window_size)
        .apply(base_flow, args=(df, info), raw=False)
    )
    df["base_flow"].iloc[0] = 0

    df["total_flow"] = df["base_flow"] + df["sup_flow"]

    df["total_sup_vol"] = (
        df["sup_flow"].sum() * 3600 / 1e6
    )  # Total volume of runoff (m3)

    df["total_infil_vol"] = (
        df["infil_flow"].sum() * 3600 / 1e6
    )  # Total volume of rain (m3)

    df["total_input_vol"] = (df["pr"] * info["area"] * 1e6 / 1000).sum() / 1e6
    df["mass_balance"] = (
        df["total_sup_vol"] + df["total_infil_vol"] - df["total_input_vol"]
    )
    df["error"] = 100 - (
        (df["total_sup_vol"] + df["total_infil_vol"]) / df["total_input_vol"] * 100
    )
    return df


def wet_soil(window, cn):
    """Determine if soil is in wet condition based on rainfall history.

    Evaluates soil moisture dynamics using a rolling window. Classifies soil as wet
    if continuous rainfall occurred within the last 5 hours.

    Parameters
    ----------
    window : np.ndarray
        Rolling window of cumulative precipitation values
    cn : float
        Curve Number value for wet soil conditions (CN3)

    Returns
    -------
    float
        Curve Number for wet conditions if all values in window are non-zero,
        otherwise returns NaN
    
    Notes
    -----
    Soil moisture classification criteria:
    
    - More than 5 hours of continuous rain: wet soil (returns cn)
    - More than 5 days without rain: dry soil (see dry_soil function)
    - Between 1-5 hours of rain: average conditions (CN2)
    """
    if np.count_nonzero(~np.isclose(window, 0)) == window.size:
        value = cn
    else:
        value = np.NaN

    return value


def dry_soil(window, cn):
    """Determine if soil is in dry condition based on antecedent dry period.

    Evaluates soil moisture dynamics using a rolling window. Classifies soil as dry
    if no rainfall occurred within the last 5 days.

    Parameters
    ----------
    window : np.ndarray
        Rolling window of cumulative precipitation values (typically 5 days = 120 hours)
    cn : float
        Curve Number value for dry soil conditions (CN1)

    Returns
    -------
    float
        Curve Number for dry conditions if all values in window are zero,
        otherwise returns NaN
    """
    if np.count_nonzero(np.isclose(window, 0)) == window.size:
        value = cn
    else:
        value = np.NaN

    return value


def unit_hydrograph_model(df, info):
    """Transform rainfall to discharge using SCS Unit Hydrograph method.

    Applies unit hydrograph theory to convert effective precipitation into
    surface runoff hydrographs. Supports multiple concentration time formulas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with effective precipitation ('net_pr' column)
    info : dict
        Dictionary with watershed parameters:
        
        - model : str
            Concentration time model ('SCS', 'Temez', 'Kirpich',
            'Kirpich-natural_slope')
        - lon : float
            Main river length (km)
        - slope : float
            Main river slope (m/m)
        - cn2 : float
            Curve Number for average conditions
        - area : float
            Watershed area (km²)
        - dt : float
            Time step (hours)

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'sup_flow' column containing surface flow (m³/s)

    Raises
    ------
    ValueError
        If the specified unit hydrograph model is not implemented
    
    Notes
    -----
    Concentration time formulas:
    
    - SCS: tc = 0.071 * L^0.8 / (S^0.25) * (1000/CN - 9)^0.7
    - Temez: tc = 0.3 * (L / S^0.25)^0.76
    - Kirpich: tc = 0.066 * L^0.77 / S^0.385
    - Kirpich-natural_slope: tc = 0.13 * L^0.77 / S^0.385
    """
    if info["model"] == "SCS":
        tc = (
            0.071
            * info["lon"] ** (0.8)
            / (info["slope"] ** (1 / 4))
            * (1000 / info["cn2"] - 9) ** (0.7)
        )
    elif info["model"] == "Temez":
        tc = 0.3 * (info["lon"] / info["slope"] ** (1 / 4)) ** (0.76)
    elif info["model"] == "Kirpich":
        tc = 0.066 * (info["lon"] ** (0.77) / info["slope"] ** (0.385))
    elif info["model"] == "Kirpich-natural_slope":
        tc = 0.13 * (info["lon"] ** (0.77) / info["slope"] ** (0.385))
    else:
        raise ValueError(
            "The unit hydrograph model given ({}) is not yet implemented".format(
                info["model"]
            )
        )
    t_p = 0.5 * info["dt"] + 0.6 * tc
    t_b = 2.67 * t_p
    q_p = info["area"] / (1.8 * t_b)

    hu1 = np.array([[0, 0], [t_p, q_p], [t_b, 0]])
    hu2 = interp1d(hu1[:, 0], hu1[:, 1])(np.arange(1, np.ceil(t_b)))

    # Desarrollamos la matriz de convolucion discreta
    df["sup_flow"] = np.convolve(df["net_pr"], hu2)[0 : len(df["net_pr"])]
    return df


def base_flow(window, df, info):
    """Compute base flow using exponential recession model.

    Calculates subsurface flow contribution to total streamflow using an
    exponential recession equation with aquifer storage dynamics.

    Parameters
    ----------
    window : pd.Series
        Rolling window containing current and previous time step
    df : pd.DataFrame
        DataFrame containing flow computations with 'base_flow' and 'infil_flow' columns
    info : dict
        Dictionary with parameters:
        
        - dt : float
            Time step (hours)
        - k_ac : float
            Aquifer storage constant (hours)

    Returns
    -------
    float
        Updated base flow value at current time step (m³/s)
    
    Notes
    -----
    Uses exponential recession:
    
    Q_base(t) = Q_base(t-1) * exp(-dt/k) + Q_infil * (1 - exp(-dt/k))
    where k is the aquifer storage constant controlling recession rate.
    """
    ex = np.exp(-info["dt"] / info["k_ac"])
    df["base_flow"].acumulado = df["base_flow"].acumulado * ex + df["infil_flow"][
        window.index[1]
    ] * (1 - ex)

    return df["base_flow"].acumulado


def distribute_precipitation(data, pattern, df, info):
    """Distribute precipitation data temporally according to SCS rainfall pattern.

    Disaggregates daily or multi-hour precipitation data into hourly values
    following standard SCS (Soil Conservation Service) rainfall distribution patterns.

    Parameters
    ----------
    data : pd.Series
        Precipitation time series with datetime index
    pattern : np.ndarray
        SCS precipitation distribution pattern (cumulative fractions)
    df : pd.DataFrame
        Empty DataFrame with hourly datetime index to fill
    info : dict
        Dictionary with parameters:
        
        - freq_raw_data : str
            Frequency of input data ('D' for daily, 'H' for hourly, 'nH' for n-hour)
        - events : bool
            Whether data represents discrete storm events

    Returns
    -------
    pd.DataFrame
        DataFrame with distributed hourly precipitation in 'pr' column
    
    Notes
    -----
    Saves distributed precipitation to 'distributed_precipitation.zip' CSV file.
    """
    if info["events"]:
        dates = []
        for ind_, _ in enumerate(data.index):
            dates.append(
                data.index[ind_]
                - pd.Timedelta(
                    seconds=data.index[ind_].minute * 60
                    + data.index[ind_].second
                    + data.index[ind_].microsecond / 1e6
                )
            )
        data.index = dates

    if info["freq_raw_data"] == "D":
        df["pr"] = np.outer(data.values, pattern).ravel()
    elif info["freq_raw_data"] == "H":
        df["pr"] = data.values
    else:
        hours = data.index.hour - 1
        k, group = 0, 0
        group_len = int(info["freq_raw_data"].split("H")[0])
        while k < len(hours):
            if group == 0:
                locs = [hours[k] + gr for gr in range(group_len)]
                indexs = np.fmod(locs, 24)
                mult_ = pattern[hours[indexs]] / np.sum(pattern[hours[indexs]])
                date = data.index[k]

            df.loc[date + pd.Timedelta(hours=group)] = data.loc[date] * mult_[group]

            if group == group_len - 1:
                group = 0
                k += 1
            else:
                group += 1

    save.to_csv(df, "distributed_precipitation.zip")
    return df


def cumulative_by_events(df, eps: float = 1e-6):
    """Compute cumulative sum of precipitation resetting between storm events.

    Calculates running cumulative precipitation that resets to zero during
    periods without rainfall, effectively separating discrete storm events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pr' column containing precipitation values
    eps : float, optional
        Threshold below which values are considered zero. Default: 1e-6

    Returns
    -------
    pd.Series
        Cumulative precipitation with resets at dry periods
    
    Notes
    -----
    This function is essential for the SCS Curve Number method as it tracks
    precipitation accumulation within individual storm events for proper
    infiltration and runoff calculations.
    """
    pr_hr = df.values[:, 0]
    cumsum = np.cumsum(pr_hr)

    cumsum_reset = cumsum[pr_hr == 0]
    cumsum_reset = np.insert(cumsum_reset, 0, 0)
    correction = np.diff(cumsum_reset)

    p_horaria_corrected = pr_hr.copy()
    p_horaria_corrected[pr_hr == 0] -= correction

    df_cumulative = pd.Series(p_horaria_corrected.cumsum(), index=df.index)
    df_cumulative[df_cumulative < eps] = 0
    return df_cumulative


def hydraulic_radius(channel_info, type_="rectangular"):
    """Calculate hydraulic properties of open channel cross-sections.

    Computes wetted area, wetted perimeter, hydraulic radius, and water surface
    width for various channel geometries used in open channel flow calculations.

    Parameters
    ----------
    channel_info : dict
        Dictionary with channel geometry parameters:
        
        - b : float
            Channel bottom width (m)
        - Additional parameters depending on channel type
            
    type_ : str, optional
        Channel cross-section type. Options: 'rectangular', 'trapezoidal',
        'triangular', 'circular', 'parabolic'. Default: 'rectangular'

    Returns
    -------
    pd.DataFrame
        DataFrame with computed hydraulic properties:
        - wet_area : Wetted cross-sectional area (m²)
        - wet_perimeter : Wetted perimeter (m)
        - rh : Hydraulic radius (m)
        - water_mirror : Water surface width (m)

    Raises
    ------
    ValueError
        If channel type is not implemented

    Notes
    -----
    Hydraulic radius is defined as: Rh = A / P
    where A is the wetted area and P is the wetted perimeter.
    This parameter is fundamental for Manning's equation and other
    open channel flow formulas.
    
    Warning
    -------
    This function has a bug: variable 'y' (water depth) is not defined as parameter
    """

    df = df.copy()

    # TODO: Check that b and y can be inside the channel_info dictionary
    if type_ == "rectangular":
        df["wet_area"] = channel_info["b"] * channel_info["y"]
        df["wet_perimeter"] = channel_info["b"] + 2 * channel_info["y"]
        df["rh"] = channel_info["b"] * channel_info["y"] / (channel_info["b"] + 2 * channel_info["y"])
        df["water_mirror"] = channel_info["b"]
    elif type_ == "trapezoidal":
        df["wet_area"] = (channel_info["b"] + channel_info["y"]) * channel_info["y"]
        df["wet_perimeter"] = channel_info["b"] + 2 * channel_info["y"]
        df["rh"] = channel_info["b"] * channel_info["y"] / (channel_info["b"] + 2 * channel_info["y"])
        df["water_mirror"] = channel_info["b"]
    elif type_ == "triangular":
        df["wet_area"] = channel_info["b"] * channel_info["y"]
        df["wet_perimeter"] = channel_info["b"] + 2 * channel_info["y"]
        df["rh"] = channel_info["b"] * channel_info["y"] / (channel_info["b"] + 2 * channel_info["y"])
        df["water_mirror"] = channel_info["b"]
    elif type_ == "circular":
        df["wet_area"] = channel_info["b"] * channel_info["y"]
        df["wet_perimeter"] = channel_info["b"] + 2 * channel_info["y"]
        df["rh"] = channel_info["b"] * channel_info["y"] / (channel_info["b"] + 2 * channel_info["y"])
        df["water_mirror"] = channel_info["b"]
    elif type_ == "parabolic":
        df["wet_area"] = channel_info["b"] * channel_info["y"]
        df["wet_perimeter"] = channel_info["b"] + 2 * channel_info["y"]
        df["rh"] = channel_info["b"] * channel_info["y"] / (channel_info["b"] + 2 * channel_info["y"])
        df["water_mirror"] = channel_info["b"]
    else:
        raise ValueError("Type of section not yet implemented")

    return df


def water_elevation(type_="rectangular"):
    """Generate objective function for water depth calculation using Manning's equation.

    Creates and returns a function that computes the residual between observed
    discharge and Manning's equation prediction for different channel geometries.
    Used for iterative water depth calculation given discharge.

    Parameters
    ----------
    type_ : str, optional
        Channel cross-section type. Options:
        - 'rectangular' : Rectangular channel
        - 'trapezoidal' : Trapezoidal channel
        - 'triangular' : Triangular channel
        - 'circular' : Circular pipe/culvert
        - 'parabolic' : Parabolic channel
        Default: 'rectangular'

    Returns
    -------
    callable
        Objective function f(h, Q, channel_params) that returns the absolute residual
        between actual discharge Q and Manning's equation prediction.
        Used with numerical solvers to find water depth h.

    Raises
    ------
    ValueError
        If channel section type is not implemented

    Notes
    -----
    Manning's equation: Q = (1/n) * A * R^(2/3) * S^(1/2)
    
    where:
    
    - Q : Discharge (m³/s)
    - n : Manning's roughness coefficient
    - A : Wetted area (m²)
    - R : Hydraulic radius (m)
    - S : Channel slope (m/m)

    The returned function should be minimized (e.g., using scipy.optimize.minimize_scalar)
    to find the water depth that produces the given discharge.

    channel_params dictionary should contain:
    
    - n : Manning's roughness coefficient
    - w : Channel width or bottom width (m)
    - S : Channel bed slope (m/m)
    - z : Side slope (for trapezoidal/triangular, horizontal:vertical)
    - D : Diameter (for circular)
    - T : Top width (for parabolic)
    - theta : Central angle (for circular)
    """
    if type_ == "rectangular":

        def rectangular(h, Q, channel_params):
            return np.abs(
                Q
                - 1
                / channel_params["n"]
                * (channel_params["w"] * h)
                * (channel_params["w"] * h / (channel_params["w"] + 2 * h)) ** (2 / 3)
                * channel_params["S"] ** 0.5
            )

        fun = rectangular
    elif type_ == "trapezoidal":

        def trapezoidal(h, Q, channel_params):
            return np.abs(
                Q
                - 1
                / channel_params["n"]
                * (channel_params["w"] * channel_params["z"] * h)
                * h
                / (channel_params["w"] + 2 * h * np.sqrt(1 + channel_params["z"] ** 2)) ** (2 / 3)
                * channel_params["S"] ** 0.5
            )

        fun = trapezoidal
    elif type_ == "triangular":

        def triangular(h, Q, channel_params):
            return np.abs(
                Q
                - 1
                / channel_params["n"]
                * (channel_params["z"] * h)
                / (2 * np.sqrt(1 + channel_params["z"] ** 2)) ** (2 / 3)
                * channel_params["S"] ** 0.5
            )

        fun = triangular
    elif type_ == "circular":

        def circular(h, Q, channel_params):
            return np.abs(
                Q
                - 1
                / channel_params["n"]
                * (
                    (1 - np.sin(np.rad2deg(channel_params["theta"])) / np.rad2deg(channel_params["theta"]))
                    * channel_params["D"]
                    / 4
                )
                ** (2 / 3)
                * channel_params["S"] ** 0.5
            )

        fun = circular
    elif type_ == "parabolic":

        def parabolic(h, Q, channel_params):
            return np.abs(
                Q
                - 1
                / channel_params["n"]
                * ((2 * channel_params["T"] ** 2 * h) / (3 * channel_params["T"] ** 2 + 8 * h**2))
                ** (2 / 3)
                * channel_params["S"] ** 0.5
            )

        fun = parabolic
    else:
        raise ValueError("Type of section not yet implemented")

    return fun


def settling_velocity(ds, info, type_="Rubey"):
    """Calculate sediment particle settling velocity in water.

    Computes the terminal fall velocity of sediment particles using empirical
    formulas that account for particle size, shape, and fluid properties.

    Parameters
    ----------
    ds : float or np.ndarray
        Sediment grain diameter (m)
    info : dict
        Dictionary with fluid and sediment properties:
        
        - nu : float
            Kinematic viscosity of water (m²/s)
        - sg : float
            Specific gravity of sediment (dimensionless, typically 2.65 for quartz)
        - Sp : float
            Sediment shape factor (required for Wu_Wang method)
            
    type_ : str, optional
        Settling velocity formula: 'Rubey' or 'Wu_Wang'. Default: 'Rubey'

    Returns
    -------
    float or np.ndarray
        Settling velocity (m/s)

    Raises
    ------
    ValueError
        If settling velocity formula is not implemented

    Notes
    -----
    Available formulas:
    
    - Rubey (1933): Valid for natural sediments across wide size range
    - Wu & Wang (2006): Accounts for particle shape effects
    """
    g = 9.81

    if type_ == "Rubey":
        F = (2 / 3 + 36 * info["nu"] ** 2 / (g * (info["sg"] - 1) * ds**3)) ** (
            1 / 2
        ) - (36 * info["nu"] ** 2 / (g * (info["sg"] - 1) * ds**3)) ** (1 / 2)
        w_s = F * (g * (info["sg"] - 1) * ds) ** (0.5)
    elif type_ == "Wu_Wang":
        M = 53.5 * np.exp(-0.65 * info["Sp"])
        N = 5.65 * np.exp(-2.5 * info["Sp"])
        n = 0.7 + 0.9 * info["Sp"]
        w_s = (
            M
            * info["nu"]
            / (N * d)
            * (np.sqrt(0.25 + (4 * N / (3 * M**2) * d_ast**3) ** (1 / n)) - 0.5) ** n
        )

    else:
        raise ValueError("Settling velocity formula not implemented.")

    return w_s


def river_sediment_transport(df, info, type_="meyer-peter-muller"):
    """Calculate river bed-load sediment transport rate using various formulas.

    Computes sediment transport in rivers using established empirical and
    semi-empirical formulas. Supports multiple methods for different grain sizes
    and flow conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hydraulic variables:
        - h : Water depth (m)
    info : dict
        Dictionary with sediment and channel properties:
        
        - rho_w : float
            Water density (kg/m³)
        - rho_s : float
            Sediment density (kg/m³)
        - sg : float
            Specific gravity of sediment (dimensionless)
        - d50 : float
            Median grain diameter (m)
        - S : float
            Channel bed slope (m/m)
        - w : float
            Channel width (m)
        - nu : float
            Kinematic viscosity (m²/s, for some methods)
        - ds_substrate : array-like
            Grain size distribution (for Wilcock-Crowe, Yang, Brownlie)
        - frac_substrate : array-like
            Fraction of each grain size class (for Wilcock-Crowe)
        - n : float
            Manning's roughness coefficient (for some methods)
            
    type_ : str, optional
        Sediment transport formula. Options:
        - 'meyer-peter-muller' : Meyer-Peter & Müller (1948) for gravel rivers
        - 'einstein-brown' : Einstein & Brown (1942) probabilistic approach
        - 'wilcock-crowe' : Wilcock & Crowe (2003) for mixed-size sediments
        - 'bagnold' : Bagnold (1966) stream power approach
        - 'yang' : Yang (1973) for sand-bed rivers
        - 'brownlie' : Brownlie (1981) comprehensive formula
        Default: 'meyer-peter-muller'

    Returns
    -------
    pd.Series or np.ndarray
        Bed-load sediment transport rate Qb (m³/hour or m³/s depending on formula)

    Notes
    -----
    Meyer-Peter-Müller formula:
    
        qb* = 8 * (τ* - τ*c)^1.5
        
        Valid for gravel-bed rivers with uniform sediment
    
    Einstein-Brown formula:
    
        Uses probability approach with three transport regimes based on τ*
    
    Wilcock-Crowe formula:
    
        Accounts for hiding/exposure effects in mixed-size sediments
        
        Requires grain size distribution input
    
    Bagnold formula:
    
        Based on stream power concept: qb ∝ τ * U
    
    Yang formula:
    
        Empirical formula for sand transport in terms of unit stream power
    
    Brownlie formula:
    
        General formula with dimensionless critical Shields parameter

    References
    ----------
    - Meyer-Peter, E., & Müller, R. (1948)
    - Einstein, H. A., & Brown, C. B. (1942)
    - Wilcock, P. R., & Crowe, J. C. (2003)
    - Bagnold, R. A. (1966)
    - Yang, C. T. (1973)
    - Brownlie, W. R. (1981)
    """

    g = 9.81

    tau = info["rho_w"] * g * df["h"] * info["S"]
    tau_shi = tau / (g * info["d50"] * info["rho_w"] * (info["sg"] - 1))
    tau_c_shi = 0.047

    if type_ == "meyer-peter-muller":
        logger.info(
            "Based on Meyer-Peter-Müller (1948) for gravel rivers with it uses d50"
            " of the superficial bed layers"
        )
        df["qb"] = 0
        motion = tau_shi >= tau_c_shi

        df["qb"][motion] = (
            8
            * (tau_shi[motion] - tau_c_shi) ** (1.5)
            * (info["d50"] * (g * (info["sg"] - 1) * info["d50"]) ** 0.5)
        )
        df["Qb"] = df["qb"] * 3600 * info["w"]

    elif type_ == "einstein-brown":

        w_s = settling_velocity(info["d50"], info)
        low_motion = tau_shi < 0.18
        mid_motion = (tau_shi > 0.18) & (tau_shi < 0.52)
        high_motion = tau_shi > 0.52
        df["qb"] = 0
        df["qb"][low_motion] = 2.15 * np.exp(-0.391 / tau_shi[low_motion])
        df["qb"][mid_motion] = 40 * tau_shi[mid_motion] ** 3
        df["qb"][high_motion] = 15 * tau_shi[high_motion] ** (1.5)

        df["qb"] = df["qb"] * w_s * info["d50"]
        df["Qb"] = df["qb"] * 3600 * info["w"]  # m3

    elif type_ == "wilcock-crowe":

        dm = np.sum(info["ds_substrate"] * info["frac_substrate"])
        if not "Fs" in info.keys():
            Fs = 0.2  # Porcentaje de arena en la capa de superficie y substrato
            logger.info("Percentage of sands if not given. Using Fs equals to 0.2.")

        tau_rm_shi = 0.021 + 0.015 * np.exp(-20 * Fs)
        tau_rm = tau_rm_shi / (g * info["d50"] * info["rho_w"] * (info["sg"] - 1))
        weight = np.zeros(len(tau))

        df["qb"] = 0

        for i in range(0, len(info["ds_substrate"]) - 1):
            b = 0.67 / (1 + np.exp(1.5 - info["ds_substrate"][i] / dm))  # bug aquí
            tau_ri = tau_rm * (info["ds_substrate"][i] / info["d50"]) ** b
            Fi = tau / tau_ri
            mask = Fi < 1.35
            weight[mask] = 0.002 * Fi[mask] ** (7.5)
            weight[~mask] = 14 * (1 - 0.894 / ((Fi[~mask]) ** 0.5)) ** 4.5

            df["qb"] = df["qb"] + weight * info["frac_substrate"][i] * info["w"] * (
                g * df["h"] * info["S"]
            ) ** (3 / 2) * info["rho_s"] / (
                (info["sg"] - 1) * g
            )  # o sobra un rho_s

        df["Qb"] = df["qb"] * 3600 / info["rho_s"]  # o falta un rho_s

    elif type_ == "bagnold":
        if not "d50" in info.keys():
            raise ValueError("d50 is not given.")

        w_s = settling_velocity(info["d50"], info)

        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        if not "eb" in info.keys():
            info["eb"] = 0.15
            logger.info("Bagnold's effitienty parameter not given. Using 0.15.")

        df["Qb"] = (
            tau
            * U
            / (info["sg"] - 1)
            * (info["eb"] + 0.01 * U / w_s)
            * info["w"]
            / info["rho_s"]
            * 3600
        )

    elif type_ == "yang":

        if not "ds_substrate" in info.keys():
            raise ValueError("ds is required for Yan method.")

        w_s = settling_velocity(info["ds_substrate"], info)

        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        Q = U * info["w"] * df["h"]
        uc = (g * df["h"] * info["S"]) ** 0.5

        landa = np.zeros((len(info["ds_substrate"]), len(h)))
        Cm2 = np.zeros((len(info["ds_substrate"]), len(h)))  # concentración másica
        Qv2 = np.zeros((len(info["ds_substrate"]), len(h)))  # caudal sólido

        for i in range(0, len(info["ds_substrate"]) - 1):
            fac1 = np.zeros(len(U))
            mask = (info["ds_substrate"][i] > 0) & (
                uc * info["ds_substrate"][i] / info["nu"]
                > 1.2 & uc * info["ds_substrate"][i] / info["nu"]
                < 70
            )
            fac1[mask] = (
                2.5 / (np.log(uc[mask] * info["ds_substrate"][i] / info["nu"]) - 0.06)
                + 0.66
            )
            mask = (info["ds_substrate"][i] > 0) & (
                uc * info["ds_substrate"][i] / info["nu"] >= 70
            )
            fac1[mask] = 2.05

            mask = (info["ds_substrate"][i] > 0) & (info["ds_substrate"][i] < 0.002)
            landa[i][mask] = (
                5.435
                - 0.286 * np.log(w_s[i] * info["ds"][i] / info["nu"])
                - 0.457 * np.log(uc[mask] / w_s[i])
            )
            +(
                1.799
                - 0.409 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 0.314 * np.log(uc[mask] / w_s[i])
            ) * np.log(U[mask] * info["S"] / w_s[i] - fac1[mask] * info["S"])

            mask = info["ds_substrate"][i] > 0.002
            landa[i][mask] = (
                6.681
                - 0.633 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 4.816 * np.log(uc[mask] / w_s[i])
            )
            +(
                2.784
                - 0.305 * np.log(w_s[i] * info["ds_substrate"][i] / info["nu"])
                - 0.282 * np.log(uc[mask] / w_s[i])
            ) * np.log(U[mask] * info["S"] / w_s[i] - fac1[mask] * info["S"])

            mask = info["ds_substrate"][i] == 0
            landa[i][mask] = 0

            Cm2[i] = 10 ** landa[i]
            Qv2[i] = Cm2[i] * 0.001 * Q / info["rho_s"]

    elif type_ == "browlie":
        U = 1 / info["n"] * df["h"] ** (2 / 3) * info["S"] ** 0.5
        Q = U * info["w"] * df["h"]

        d_adim = info["ds_substrate"] * ((info["sg"] - 1) * g / (info["nu"] ** 2)) ** (
            1 / 3
        )

        if "tau_adim" in info:
            if info["tau_adim"] == "Wu_Wang":
                tau_adim_c = np.zeros((len(d_adim)))
                mask = d_adim <= 1.5
                tau_adim_c[mask] = 0.126 * d_adim[mask] ** (-0.44)
                mask = (d_adim > 1.5) & (d_adim <= 10)
                tau_adim_c[mask] = 0.131 * d_adim[mask] ** (-0.55)
                mask = (d_adim > 10) & (d_adim <= 20)
                tau_adim_c[mask] = 0.0685 * d_adim[mask] ** (-0.27)
                mask = (d_adim > 20) & (d_adim <= 40)
                tau_adim_c[mask] = 0.0173 * d_adim[mask] ** (0.19)
                mask = (d_adim > 40) & (d_adim <= 150)
                tau_adim_c[mask] = 0.0115 * d_adim[mask] ** (0.30)
                mask = d_adim > 150
                tau_adim_c[mask] = 0.052
        else:
            logger.info("Browlie's tau adimensional parameter.")
            dens_apa = 1600
            Y_bw = (
                g**0.5
                * (dens_apa / info["rho_w"]) ** 0.5
                * info["d50"] ** (3 / 2)
                / info["nu"]
            ) ** (-0.6)
            tau_adim_c = 0.22 * Y_bw + 0.06 * 10 ** (7.7 * Y_bw)

        df["Cppm"] = 0
        df["Qb"] = 0
        cb = 1.268

        Uc = (
            ((info["sg"] - 1) * g * info["d50"]) ** 0.5
            * 4.596
            * tau_adim_c**0.529
            * info["S"] ** (-0.1405)
            * np.std(info["ds_substrate"]) ** (-0.1606)
        )
        df["Cppm"] = (
            7115
            * cb
            * ((U - Uc) / ((info["sg"] - 1) * g * info["d50"]) ** 0.5) ** 1.978
            * info["S"] ** 0.6601
            * (df["h"] / info["d50"]) ** (-0.3301)
        )
        df["Qb"] = df["Cppm"] * 0.001 * Q * 3600 / info["rho_s"]

    return df["Qb"]


def storm_surge_from_waves(data: pd.DataFrame, location: str, var_name: str = "Hm0"):
    """Compute storm surge elevation from significant wave height.

    Estimates storm surge (sea level anomaly) based on wave conditions using
    empirical relationships from the Spanish Flooding Atlas (Atlas de Inundación Español).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with wave parameters
    location : str
        Location identifier for parameter selection. Options: 'Huelva', 'Malaga'
    var_name : str, optional
        Column name for significant wave height (m). Default: 'Hm0'

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'mm' column containing storm surge elevation (m)

    Notes
    -----
    Uses location-specific polynomial relationships between wave height and surge:
    
    - Location parameter (mu): mean surge level
    - Scale parameter (sigma): surge variability
    
    Storm surge is computed stochastically using normal distribution with
    wave-height-dependent parameters.

    References
    ----------
    Atlas de Inundación de la Costa Española
    """

    parametros = {
        "Malaga": {
            "loc": [-0.0334122, 0.08969, -0.0148889],
            "esc": [0.0875315, -0.0238893, 0.0473336, -0.0122261],
        },
        "Huelva": {
            "loc": [-0.0218071, -0.00208886, 0.0286215],
            "esc": [0.0558449, 0.0446334, -0.0106856, -0.000660628],
        },
        "info": "Parametros del Atlas de Inundacion",
    }

    mu = np.polyval(parametros[location]["loc"][::-1], data[var_name])
    esc = np.polyval(parametros[location]["esc"][::-1], data[var_name])

    # Replace negative variance
    esc[esc < 0] = np.random.rand(len(esc[esc < 0])) * 1e-3

    p_Hs = np.random.rand(len(data))
    eta = st.norm.ppf(p_Hs, mu, esc)
    data["mm"] = data[var_name] * 0 + eta
    return data


def flood_fill(c, r, mask):
    """Perform flood fill algorithm on a binary mask to identify connected regions.

    Implements a non-recursive 8-way connectivity flood fill algorithm. Starting
    from a seed cell, identifies all connected cells with value 1 in the mask.
    Useful for delineating inundation areas or connected coastal regions.

    Parameters
    ----------
    c : int
        Starting column index (x-coordinate)
    r : int
        Starting row index (y-coordinate)
    mask : np.ndarray
        Binary 2D array containing only 1 and 0 values, where 1 represents
        cells to potentially fill

    Returns
    -------
    np.ndarray
        Binary array of same shape as mask, with 1 values for all cells
        connected to the starting point, 0 elsewhere

    Notes
    -----
    This algorithm uses 8-way connectivity (includes diagonal neighbors).
    It's non-recursive to avoid stack overflow issues with large connected regions.
    
    The algorithm is commonly used for:
    
    - Identifying inundation extents from water level data
    - Finding connected coastal segments
    - Delineating watersheds or drainage basins
    
    Complexity: O(n) where n is the number of connected cells
    """
    # cells already filled
    filled = set()
    # cells to fill
    fill = set()
    fill.add((c, r))
    width = mask.shape[1] - 1
    height = mask.shape[0] - 1
    # Our output inundation array
    flood = np.zeros_like(mask, dtype=np.int8)
    # Loop through and modify the cells which need to be checked.
    while fill:
        # Grab a cell
        x, y = fill.pop()
        if (x <= height) & (y <= width) & (x >= 0) & (y >= 0):
            # Don't fill
            # continue
            if (mask[x][y] == 1) & ((x, y) not in filled):
                # Do fill
                flood[x][y] = 1
                filled.add((x, y))
                # Check neighbors for 1 values
                west = (x - 1, y)
                east = (x + 1, y)
                north = (x, y - 1)
                south = (x, y + 1)
                northwest = (x - 1, y - 1)
                northeast = (x + 1, y - 1)
                southwest = (x - 1, y + 1)
                southeast = (x + 1, y + 1)
                if west not in filled:
                    fill.add(west)
                if east not in filled:
                    fill.add(east)
                if north not in filled:
                    fill.add(north)
                if south not in filled:
                    fill.add(south)
                if northwest not in filled:
                    fill.add(northwest)
                if northeast not in filled:
                    fill.add(northeast)
                if southwest not in filled:
                    fill.add(southwest)
                if southeast not in filled:
                    fill.add(southeast)
    return flood


def EOS_sea_water(T, S):
    """Compute density of seawater using equation of state (EOS).
    
    Calculates seawater density as a function of temperature and salinity
    using the standard mean ocean water equation of state with pure water
    as reference. Includes pressure correction at 1 meter depth.

    Parameters
    ----------
    T : pd.DataFrame or np.ndarray
        Water temperature (°C)
    S : pd.DataFrame or np.ndarray
        Salinity (psu - practical salinity units)

    Returns
    -------
    pd.DataFrame or np.ndarray
        Seawater density (kg/m³) at given temperature and salinity
        
    Notes
    -----
    The calculation follows the standard seawater equation of state:
    
    1. Computes pure water density as function of temperature
    2. Adjusts for salinity effects
    3. Applies high pressure correction (p = 0.10073 bar at 1m depth)
    
    The equation uses empirical polynomial coefficients derived from
    oceanographic measurements and is valid for typical ocean conditions.

    Examples
    --------
    >>> import pandas as pd
    >>> T = pd.Series([15, 20, 25])  # Temperature in °C
    >>> S = pd.Series([35, 35, 35])  # Salinity in psu
    >>> rho = EOS_sea_water(T, S)
    >>> print(rho)
    """
    p = 0.10073  # bar at -1m
    rho_w = (
        999.842594
        + 6.793952 * 1e-2 * T
        - 9.095290 * 1e-3 * T**2
        + 1.001685 * 1e-4 * T**3
        - 1.120083 * 1e-6 * T**4
        + 6.536332 * 1e-9 * T**5
    )

    # Density of sea water at one standard atmosphere (p=0)
    rho_S_T_0 = (
        rho_w
        + (
            8.24493e-1
            - 4.0899e-3 * T
            + 7.6438e-5 * T**2
            - 8.2467e-7 * T**3
            + 5.3875e-9 * T**4
        )
        * S
        + (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2) * (S ** (3 / 2))
        + 4.8314e-4 * S**2
    )

    # Density of sea water at high pressure is: rho_S_T_p
    Kw = (
        19652.21
        + 148.4206 * T
        - 2.327105 * T**2
        + 1.360477e-2 * T**3
        - 5.155288e-5 * T**4
    )
    K_S_T_0 = (
        Kw
        + (54.6746 - 0.603459 * T + 1.09987e-2 * T**2 - 6.1670e-5 * T**3) * S
        + (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T**2) * S ** (3 / 2)
    )

    Aw = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T**2 - 5.77905e-7 * T**3
    A = (
        Aw
        + (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T**2) * S
        + 1.91075e-4 * S ** (3 / 2)
    )
    Bw = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T**2
    B = Bw + (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T**2) * S
    K_S_T_p = K_S_T_0 + A * p + B * p**2
    rho = rho_S_T_0 / (1 - p / (K_S_T_p))
    return rho


def bulk_fluid_density(T, S, C, rhos=2650):
    """Compute bulk fluid density from suspended sediment concentration.
    
    Calculates the density of water-sediment mixture accounting for
    temperature, salinity, and suspended sediment concentration (turbidity).
    
    Parameters
    ----------
    T : pd.DataFrame or np.ndarray
        Water temperature (°C)
    S : pd.DataFrame or np.ndarray
        Salinity (psu - practical salinity units)
    C : pd.DataFrame or np.ndarray
        Turbidity, suspended sediment concentration (FNU - Formazin Nephelometric Units)
    rhos : float, optional
        Sediment particle density (kg/m³). Default: 2650 (typical quartz sand)

    Returns
    -------
    rho : pd.DataFrame or np.ndarray
        Clear water density (kg/m³) from temperature and salinity
    rho_bulk : pd.DataFrame or np.ndarray
        Bulk fluid density (kg/m³) including suspended sediment effects
        
    Notes
    -----
    Conversion factor: Multiply turbidity C by 1.6015e-3 to convert FNU to g/L.
    
    The bulk density accounts for:
    
    - Base seawater density from EOS (temperature and salinity effects)
    - Volume displacement by sediment particles
    - Mass contribution from suspended sediment
    
    Formula: ρ_bulk = ρ_water + (1 - ρ_water/ρ_sediment) * C * 1.6015e-3

    Examples
    --------
    >>> import pandas as pd
    >>> T = pd.Series([20, 20, 20])
    >>> S = pd.Series([35, 35, 35])
    >>> C = pd.Series([0, 100, 500])  # Turbidity in FNU
    >>> rho, rho_bulk = bulk_fluid_density(T, S, C)
    >>> print(f"Clear water: {rho[0]:.2f} kg/m³")
    >>> print(f"With sediment: {rho_bulk[2]:.2f} kg/m³")
    """
    rho = EOS_sea_water(T, S)
    rho0_prof_1 = rho + (1 - (rho / rhos)) * C * 1.6015 * 1e-3
    return rho, rho0_prof_1
