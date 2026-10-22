"""Coastal processes: bottom slopes, sediment transport, plan shape and shoreline evolution."""

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.special import erfc

from environmentaltools.common import read, utils
from environmentaltools.processes import waves


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


def kobayashi_transport(depth, setup, kp, Qb, U, DirU, Sb, Hs, Tp, params):
    """Suspended and bed-load transport following Kobayashi et al. (2008).

    Pure function of 2-D fields: they can come from any wave and current model
    (e.g. SWAN + COPLA in :mod:`environmentaltools.models.swan_copla`).

    Args:
        depth (np.ndarray): Still water depth (m).
        setup (np.ndarray): Wave setup (m).
        kp (np.ndarray): Wave number (rad/m).
        Qb (np.ndarray): Fraction of breaking waves.
        U (np.ndarray): Current speed (m/s).
        DirU (np.ndarray): Current direction (degrees).
        Sb (np.ndarray): Bottom slope in the current direction
            (see :func:`slopes`).
        Hs (float): Significant wave height (m).
        Tp (float): Peak period (s).
        params (dict): Constants:

            - fb: bottom friction factor
            - wf: sediment fall velocity (m/s)
            - gamma: breaker ratio
            - rho: water density (kg/m3)
            - br: roller front slope (rad)
            - eb, ef: suspension efficiencies due to breaking and to bottom friction
            - S: specific gravity of the sediment
            - aK: suspended load parameter
            - d50: median grain size (m)
            - Shic: critical Shields parameter
            - bK: bed load parameter
            - phi: sediment friction angle (rad)
            - local_angle: grid orientation (degrees)
            - tburst: duration of the sea state (s)

    Returns:
        dict: Fields with the same shape as ``depth``: total depth ``D``,
        celerity ``cp``, ``sigma``, ``Rs``, ``Ps``, ``Hb``, roller ``Dr``,
        ``arot``, dissipations ``Db`` and ``Df``, ``Vs``, suspended load
        ``Qst``, ``Rb``, ``Pb``, ``Gs``, bed load ``Qbt`` and the transport
        ``qc`` and ``ql`` (``(Qbt + Qst) * tburst`` projected with ``DirU -
        local_angle``).
    """
    G = 9.8091
    U, DirU, Sb = np.asarray(U, float), np.asarray(DirU, float), np.asarray(Sb, float)
    kp, Qb = np.asarray(kp, float), np.asarray(Qb, float)
    f = {}
    f["D"] = np.asarray(depth, float) + np.asarray(setup, float)
    f["D"][f["D"] < 0.1] = np.nan  # Total depth
    f["cp"] = np.sqrt(G * f["D"])  # Shallow water waves
    f["sigma"] = Hs / np.sqrt(8) / f["D"] * f["cp"]
    f["sigma"][np.isnan(f["sigma"])] = 0

    f["Rs"] = (2 / params["fb"]) ** (1 / 3) * params["wf"] / f["sigma"]
    f["Rs"][np.isnan(f["Rs"])] = 0
    f["Ps"] = 0.5 * erfc((f["Rs"] + U) / np.sqrt(2)) + 0.5 * erfc(
        (f["Rs"] - U) / np.sqrt(2)
    )

    f["Hb"] = (
        0.88
        / kp
        * np.tanh(
            params["gamma"]
            * kp
            * f["D"]
            / (0.88 * f["cp"] * Tp)
        )
    )  # Kobayashi 0.88
    f["Hb"][f["Hb"] / f["D"] > 0.88] = (
        0.88 * f["D"][f["Hb"] / f["D"] > 0.88]
    )
    f["Hb"][np.isnan(f["Hb"])] = 0

    #   Suspended load transport
    # --------------------------------------------------------------------------
    f["Dr"] = (
        G
        * (params["rho"] * 0.9 * f["Hb"] ** 2 * f["cp"] / Tp)
        * np.sin(params["br"])
    )  # Roller effect
    f["Dr"][np.isnan(f["Dr"])] = 0

    f["arot"] = 1 / 3 * Sb * Tp * np.sqrt(G / f["D"])
    f["arot"][np.isnan(f["arot"])] = 0
    f["Db"] = (
        params["rho"]
        * f["arot"]
        * G
        * Qb
        * f["Hb"] ** 2
        / (4 * Tp)
    )  # Battjes and Stive, 1985
    f["Df"] = 0.5 * params["rho"] * G * params["fb"] * U ** 3
    f["Vs"] = (
        (params["eb"] * (f["Db"] - f["Dr"]) + params["ef"] * f["Df"])
        * f["Ps"]
        / (params["rho"] * G * (params["S"] - 1) * params["wf"])
    )
    f["Qst"] = params["aK"] * U * f["Vs"] * np.sqrt(1 + Sb ** 2)

    #    Bed-load transport
    # --------------------------------------------------------------------------
    f["Rb"] = (
        np.sqrt(
            2 * G * (params["S"] - 1) * params["d50"] * params["Shic"] / params["fb"]
        )
        / f["sigma"]
    )
    f["Rb"][np.isnan(f["Rb"])] = 0
    f["Pb"] = 0.5 * erfc((f["Rb"] + U) / np.sqrt(2)) + 0.5 * erfc(
        (f["Rb"] - U) / np.sqrt(2)
    )

    f["Gs"] = (np.tan(params["phi"]) - 2 * Sb) / (
        np.tan(params["phi"]) - Sb
    )
    f["Gs"][Sb < 0] = np.tan(params["phi"]) / (
        np.tan(params["phi"]) + Sb[Sb < 0]
    )
    f["Qbt"] = (
        params["bK"]
        * f["Pb"]
        * f["Gs"]
        * f["sigma"] ** 3
        / (G * (params["S"] - 1))
    )  # Formulación de Meyer-Peter-Mueller, modificado

    # Cross-shore and alongshore sediment transport
    # --------------------------------------------------------------------------
    f["qc"] = (
        (f["Qbt"] + f["Qst"])
        * np.sin((DirU - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][0])
    f["ql"] = (
        (f["Qbt"] + f["Qst"])
        * np.cos((DirU - params["local_angle"]) * np.pi / 180)
        * params["tburst"]
    )  # /((1.00-param['materiales']['p'])*param['malla']['local']['inc'][1])

    return f


def sediment_transport_Kobayashi(j, data, grid, params):
    """Kobayashi et al. (2008) transport on a ``grid`` dict (deprecated).

    .. deprecated:: Use :func:`kobayashi_transport`, which takes the fields
       explicitly. This wrapper reads them from ``grid`` and ``Hs``/``Tp`` from
       ``data.loc[j]``.

    Args:
        j: Index label of the sea state in ``data``.
        data (pd.DataFrame): Sea states with ``Hs`` and ``Tp``.
        grid (dict): Fields ``depth``, ``Setup``, ``kp``, ``Qb``, ``U``, ``DirU`` and ``Sb``.
        params (dict): Constants of :func:`kobayashi_transport`.

    Returns:
        dict: ``grid`` updated with the fields of :func:`kobayashi_transport`.
    """
    warnings.warn("environmentaltools.processes.sediment_transport_Kobayashi is deprecated and will be removed "
                  "in version 4.0; use environmentaltools.processes.kobayashi_transport instead.",
                  DeprecationWarning, stacklevel=2)
    grid.update(kobayashi_transport(grid["depth"], grid["Setup"], grid["kp"], grid["Qb"], grid["U"], grid["DirU"],
                                    grid["Sb"], data.loc[j, "Hs"], data.loc[j, "Tp"], params))
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
        as values. Each DataFrame should contain 'x' and 'y' columns. The first
        key is the reference coastline.
    points : list
        List of 2D points (x, y tuples or arrays) where coastline evolution is tracked

    Returns
    -------
    pd.DataFrame
        DataFrame with coastline evolution distances at each monitoring point
        (rows: time steps after the reference; columns: points). The distance is
        negative when the nearest coastline point moved towards larger y.
    """

    times = list(coastlines.keys())
    reference = coastlines[times[0]]
    evolution = pd.DataFrame(np.nan, index=times[1:], columns=range(len(points)))

    for ind_, point in enumerate(points):
        index_0 = utils.find_nearest_point(reference, point)
        x0, y0 = reference["x"].iloc[index_0], reference["y"].iloc[index_0]
        for time_ in times[1:]:
            coastline = coastlines[time_]
            index_ = utils.find_nearest_point(coastline, [x0, y0])
            x, y = coastline["x"].iloc[index_], coastline["y"].iloc[index_]
            constant = -1 if (y - y0) > 0 else 1
            evolution.loc[time_, ind_] = constant * np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    return evolution


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
