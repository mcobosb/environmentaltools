"""Coastal processes: bottom slopes, sediment transport, plan shape and shoreline evolution."""

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.special import erf, erfc

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
        Change of ``depth`` over 1 m in the specified direction (positive when the
        water gets deeper, i.e. when the bed descends). The bed slope used by
        :func:`kobayashi_transport` is ``-slopes(grid, "DirM")``.
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


_HERMITE_X, _HERMITE_W = np.polynomial.hermite_e.hermegauss(64)
_HERMITE_W = _HERMITE_W / np.sqrt(2 * np.pi)


def _friction_factor_gf(w_sta, v_sta):
    """``GF = E[((W* + ξ)² + V*²)^(3/2)]`` with ξ ~ N(0, 1).

    Mean of the cubed velocity normalized by σT, used in the bottom friction
    dissipation. For ``V* = 0`` it is the analytical expression of CSHORE
    (``GBXAGF``, normal incidence); otherwise it is computed by Gauss-Hermite
    quadrature (CSHORE uses an approximation for oblique waves).
    """
    w, v = np.broadcast_arrays(np.asarray(w_sta, float), np.asarray(v_sta, float))
    normal = w * (w**2 + 3) * erf(w / np.sqrt(2)) + (w**2 + 2) * np.sqrt(2 / np.pi) * np.exp(-(w**2) / 2)
    oblique = np.sum(_HERMITE_W * ((w[..., None] + _HERMITE_X) ** 2 + v[..., None] ** 2) ** 1.5, axis=-1)
    return np.where(np.abs(v) < 1e-12, normal, oblique)


def _probability(r, w_sta, v_sta):
    """Probability of sediment movement (``r = Rb``) or suspension (``r = Rs``), CSHORE ``SEDTRA``."""
    f = r**2 - v_sta**2
    with np.errstate(invalid="ignore"):
        f = np.sqrt(np.where(f > 0, f, 0.0))
    p = 0.5 * (erfc((f + w_sta) / np.sqrt(2)) + erfc((f - w_sta) / np.sqrt(2)))
    return np.where(r**2 - v_sta**2 < 0, 1.0, p)


def kobayashi_transport(depth, setup, Hs, Tp, kp, wave_dir, Qb, U, current_dir, Sb, params):
    """Suspended and bed-load transport of Kobayashi et al. (2008), as implemented in CSHORE.

    Pure function of 2-D fields: they can come from any wave and current model
    (e.g. SWAN + COPLA in :mod:`environmentaltools.models.swan_copla`). The
    equations follow the subroutine ``SEDTRA`` of CSHORE
    (https://github.com/erdc/cshore) for obliquely incident waves, applied at
    each node in a frame aligned with the local wave direction (``θ = 0``): the
    current is split into the component along the waves (``W*``) and across
    them (``V*``), both normalized by the standard deviation of the orbital
    velocity ``σT``.

    Adaptations to a 2-D grid (not in CSHORE, which works on cross-shore
    profiles): the bed slope ``Sb`` is taken along the wave direction; the
    friction factor ``GF`` is computed exactly for any ``V*`` (CSHORE uses an
    approximation for oblique waves); the breaking fraction ``Qb`` comes from
    the wave model; no roller (``IROLL = 0``) and no cap of the cross-shore
    undertow (``USTA <= 1``).

    Args:
        depth (np.ndarray): Still water depth (m, positive downwards).
        setup (np.ndarray): Wave setup (m).
        Hs (float | np.ndarray): Local significant wave height (m); ``Hrms = Hs/√2``.
        Tp (float | np.ndarray): Peak period (s).
        kp (np.ndarray): Wave number (rad/m).
        wave_dir (np.ndarray): Wave propagation direction (degrees).
        Qb (np.ndarray): Fraction of breaking waves.
        U (np.ndarray): Depth-averaged current speed (m/s).
        current_dir (np.ndarray): Current direction (degrees, same convention as ``wave_dir``).
        Sb (np.ndarray): Bed slope along the wave direction, positive when the bed
            rises in the direction of propagation (``-slopes(grid, "DirM")``).
        params (dict): Constants (CSHORE name in brackets):

            - fb: bottom friction factor (``fw``; CSHORE uses ``fb/2``)
            - wf: sediment fall velocity (m/s) (``WF``)
            - gamma: breaker ratio (``GAMMA``)
            - eb, ef: suspension efficiencies due to breaking and to bottom friction (``EFFB``, ``EFFF``)
            - S: specific gravity of the sediment (``SG``)
            - aK: suspended load parameter (``SLP``)
            - d50: median grain size (m) (``D50``)
            - Shic: critical Shields parameter (``SHIELD``)
            - bK: bed load parameter (``BLP``)
            - phi: sediment friction angle (rad) (``atan(TANPHI)``)
            - local_angle: grid orientation (degrees) for ``qc``/``ql``
            - tburst: duration of the sea state (s)

    Returns:
        dict: Fields with the shape of ``depth``: total depth ``D``, phase
        celerity ``cp``, orbital velocity ``sigma`` (σT), normalized currents
        ``W_sta`` and ``V_sta``, ``Rb``, ``Rs``, probabilities ``Pb`` and ``Ps``,
        breaker height ``Hb``, ``arot`` (``ABREAK``), dissipations ``Db`` and
        ``Df`` (W/m²), ``GF``, ``Gs``, ``Vs`` (m), bed load ``Qbt`` and suspended
        load ``Qst`` (m²/s, magnitude), the transport vector ``qx``, ``qy``
        (m²/s, axes of ``wave_dir``) and ``qc``, ``ql`` (m³/m over ``tburst``,
        across and along ``local_angle``).

    References:
        Kobayashi, N., Payo, A. and Schmied, L. (2008). Cross-shore suspended
        sand and bedload transport on beaches. J. Geophys. Res., 113, C07001.
    """
    G = 9.8091
    rho = params.get("rho", 1025.0)
    tanphi = np.tan(params["phi"])
    as_f = lambda a: np.asarray(a, float)  # noqa: E731
    Hs, Tp, kp, Qb, U, Sb = as_f(Hs), as_f(Tp), as_f(kp), as_f(Qb), as_f(U), as_f(Sb)
    relative = np.deg2rad(as_f(current_dir) - as_f(wave_dir))

    f = {}
    f["D"] = as_f(depth) + as_f(setup)
    f["D"] = np.where(f["D"] < 0.1, np.nan, f["D"])  # dry or very shallow nodes
    f["cp"] = 2 * np.pi / (Tp * kp)  # linear phase celerity (CP = ω/k)

    # Orbital velocity: σ_η = Hrms/√8, σ* = σ_η/h <= 1, σT = Cp σ*  (l. 424, 749-750, 996)
    sig_sta = np.minimum(Hs / np.sqrt(2) / np.sqrt(8) / f["D"], 1.0)
    f["sigma"] = f["cp"] * sig_sta

    # Current along (W*) and across (V*) the waves, normalized by σT  (l. 1006-1013, 4757-4758)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_sta = np.where(f["sigma"] > 1e-10, U / f["sigma"], 0.0)
    f["W_sta"] = u_sta * np.cos(relative)
    f["V_sta"] = u_sta * np.sin(relative)

    # Movement and suspension probabilities  (l. 4752-4780)
    fb2 = 0.5 * params["fb"]
    gsd50s = G * (params["S"] - 1) * params["d50"] * params["Shic"]
    f["Rb"] = np.sqrt(gsd50s / fb2) / f["sigma"]
    f["Rs"] = params["wf"] / f["sigma"] / fb2 ** (1 / 3)
    f["Pb"] = _probability(f["Rb"], f["W_sta"], f["V_sta"])
    f["Ps"] = np.minimum(_probability(f["Rs"], f["W_sta"], f["V_sta"]), f["Pb"])

    # Breaking dissipation  (DBREAK, l. 3104-3146)
    Hrms = Hs / np.sqrt(2)
    f["Hb"] = 0.88 / kp * np.tanh(params["gamma"] * kp * f["D"] / 0.88)
    f["Hb"] = np.maximum(f["Hb"], Hrms)  # HM = WHRMS when Hrms >= HM
    f["arot"] = np.maximum(2 * np.pi / kp / f["D"] * Sb / 3, 1.0)
    db_sta = 0.25 * f["arot"] * Qb * f["Hb"] ** 2 / Tp  # Db/(ρg)
    f["Db"] = rho * G * db_sta

    # Bottom friction dissipation  (l. 670, GBXAGF)
    f["GF"] = _friction_factor_gf(f["W_sta"], f["V_sta"])
    df_sta = fb2 * f["GF"] * f["sigma"] ** 3 / G  # Df/(ρg)
    f["Df"] = rho * G * df_sta

    # Suspended sediment volume per unit area  (l. 4783-4786)
    f["Vs"] = f["Ps"] * (params["ef"] * df_sta + params["eb"] * db_sta) / (params["wf"] * (params["S"] - 1))
    f["Vs"] = f["Vs"] * np.sqrt(1 + Sb**2)

    # Bed slope function Gs, bounded by GSLMAX = 10  (l. 4306-4334)
    gslmax = 10.0
    slope_down = -tanphi * (gslmax - 1) / gslmax
    slope_up = tanphi * (gslmax + 1) / (gslmax + 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["Gs"] = np.where(
            Sb < 0,
            np.where(Sb > slope_down, tanphi / (tanphi + Sb), gslmax),
            np.where(Sb < slope_up, (tanphi - 2 * Sb) / (tanphi - Sb), -gslmax),
        )
    a_slope = params["aK"] + np.where(Sb > 0, np.sqrt(np.abs(Sb) / tanphi), 0.0)

    # Bed load, oblique waves with θ = 0  (l. 4789-4806); BQ = BLP/(g(s-1))·(0.5 + Qb) for sand
    bq = params["bK"] / (G * (params["S"] - 1)) * (0.5 + Qb)
    dum = bq * f["Pb"] * f["sigma"] ** 3
    w, v = f["W_sta"], f["V_sta"]
    qb_along = dum * f["Gs"] * (1 + w * v**2)
    qb_across = dum * v * (1 + w**2 + v**2)

    # Suspended load: mean current × Vs  (l. 4825, 4833)
    qs_along = a_slope * U * np.cos(relative) * f["Vs"]
    qs_across = U * np.sin(relative) * f["Vs"]

    f["Qbt"] = np.hypot(qb_along, qb_across)
    f["Qst"] = np.hypot(qs_along, qs_across)

    # From the wave frame to the axes of wave_dir, and to the grid (local_angle)
    theta = np.deg2rad(as_f(wave_dir))
    q_along, q_across = qb_along + qs_along, qb_across + qs_across
    f["qx"] = q_along * np.cos(theta) - q_across * np.sin(theta)
    f["qy"] = q_along * np.sin(theta) + q_across * np.cos(theta)
    alpha = np.deg2rad(params["local_angle"])
    f["ql"] = (f["qx"] * np.cos(alpha) + f["qy"] * np.sin(alpha)) * params["tburst"]
    f["qc"] = (-f["qx"] * np.sin(alpha) + f["qy"] * np.cos(alpha)) * params["tburst"]

    return f


def sediment_transport_Kobayashi(j, data, grid, params):
    """Kobayashi et al. (2008) transport on a ``grid`` dict (deprecated).

    .. deprecated:: Use :func:`kobayashi_transport`, which takes the fields
       explicitly. This wrapper reads them from ``grid`` (``depth``, ``Setup``,
       ``Hs``, ``kp``, ``DirM``, ``Qb``, ``U``, ``DirU``, ``Sb``) and ``Tp`` from
       ``data.loc[j]``. Since the correction of the formulation (CSHORE), ``Sb``
       is the bed slope along the wave direction (positive upwards) and the
       results differ from earlier versions.

    Args:
        j: Index label of the sea state in ``data``.
        data (pd.DataFrame): Sea states with ``Tp`` (and ``Hs`` if not in ``grid``).
        grid (dict): Fields of the wave and current models.
        params (dict): Constants of :func:`kobayashi_transport`.

    Returns:
        dict: ``grid`` updated with the fields of :func:`kobayashi_transport`.
    """
    warnings.warn("environmentaltools.processes.sediment_transport_Kobayashi is deprecated and will be removed "
                  "in version 4.0; use environmentaltools.processes.kobayashi_transport instead.",
                  DeprecationWarning, stacklevel=2)
    Hs = grid["Hs"] if "Hs" in grid else data.loc[j, "Hs"]
    grid.update(kobayashi_transport(grid["depth"], grid["Setup"], Hs, data.loc[j, "Tp"], grid["kp"], grid["DirM"],
                                    grid["Qb"], grid["U"], grid["DirU"], grid["Sb"], params))
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
