"""Hydrology and river hydraulics: rainfall-runoff, unit hydrograph, channel flow and sediment transport."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import interp1d

from environmentaltools.common import read, save
from environmentaltools.processes.properties import settling_velocity


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


def _section_geometry(y, section, params):
    """Wetted area, wetted perimeter and top width of an open channel section.

    Args:
        y (float | np.ndarray): Water depth (m).
        section (str): 'rectangular', 'trapezoidal', 'triangular', 'circular' or 'parabolic'.
        params (dict): Bottom width ``b`` (or ``w``), side slope ``z`` (horizontal:vertical),
            diameter ``D`` or top width ``T``, depending on the section.

    Returns:
        tuple: ``(area, perimeter, top_width)``.
    """
    y = np.asarray(y, dtype=float)
    b = params["b"] if "b" in params else params.get("w")
    if section == "rectangular":
        return b * y, b + 2 * y, b + 0 * y
    if section == "trapezoidal":
        z = params["z"]
        return (b + z * y) * y, b + 2 * y * np.sqrt(1 + z**2), b + 2 * z * y
    if section == "triangular":
        z = params["z"]
        return z * y**2, 2 * y * np.sqrt(1 + z**2), 2 * z * y
    if section == "circular":
        D = params["D"]
        theta = 2 * np.arccos(1 - 2 * np.clip(y, 0, D) / D)  # central angle of the wetted sector
        return (theta - np.sin(theta)) * D**2 / 8, theta * D / 2, D * np.sin(theta / 2)
    if section == "parabolic":
        T = params["T"]
        return 2 / 3 * T * y, T + 8 * y**2 / (3 * T), T + 0 * y
    raise ValueError("Type of section not yet implemented")


def hydraulic_radius(channel_info, type_="rectangular"):
    """Calculate hydraulic properties of open channel cross-sections.

    Computes wetted area, wetted perimeter, hydraulic radius, and water surface
    width for various channel geometries used in open channel flow calculations.

    Parameters
    ----------
    channel_info : dict
        Dictionary with channel geometry parameters:

        - y : float or array-like
            Water depth (m)
        - b : float
            Channel bottom width (m; rectangular and trapezoidal)
        - z : float
            Side slope, horizontal:vertical (trapezoidal and triangular)
        - D : float
            Diameter (m; circular)
        - T : float
            Top width (m; parabolic)

    type_ : str, optional
        Channel cross-section type. Options: 'rectangular', 'trapezoidal',
        'triangular', 'circular', 'parabolic'. Default: 'rectangular'

    Returns
    -------
    pd.DataFrame
        One row per water depth with columns:
        - y : Water depth (m)
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
    """
    y = np.atleast_1d(np.asarray(channel_info["y"], dtype=float))
    area, perimeter, top_width = _section_geometry(y, type_, channel_info)

    df = pd.DataFrame({"y": y})
    df["wet_area"] = area
    df["wet_perimeter"] = perimeter
    df["rh"] = area / perimeter
    df["water_mirror"] = top_width
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
    - w : Channel bottom width (m; rectangular and trapezoidal)
    - S : Channel bed slope (m/m)
    - z : Side slope (for trapezoidal/triangular, horizontal:vertical)
    - D : Diameter (for circular)
    - T : Top width (for parabolic)
    """
    if type_ not in ("rectangular", "trapezoidal", "triangular", "circular", "parabolic"):
        raise ValueError("Type of section not yet implemented")

    def fun(h, Q, channel_params):
        area, perimeter, _ = _section_geometry(h, type_, channel_params)
        return np.abs(
            Q
            - 1
            / channel_params["n"]
            * area
            * (area / perimeter) ** (2 / 3)
            * channel_params["S"] ** 0.5
        )

    fun.__name__ = type_
    return fun


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
            Fraction of each grain size class (for Wilcock-Crowe and Yang)
        - Fs : float
            Fraction of sand in the surface layer (Wilcock-Crowe, default 0.2)
        - sigma_g : float
            Geometric standard deviation of the bed material (Brownlie, default 1)
        - n : float
            Manning's roughness coefficient (for some methods)
            
    type_ : str, optional
        Sediment transport formula. Options:
        - 'meyer-peter-muller' : Meyer-Peter & Müller (1948) for gravel rivers
        - 'einstein-brown' : Einstein & Brown (1942) probabilistic approach
        - 'wilcock-crowe' : Wilcock & Crowe (2003) for mixed-size sediments
        - 'bagnold' : Bagnold (1966) stream power approach
        - 'yang' : Yang (1973) for sand-bed rivers
        - 'brownlie' : Brownlie (1981) comprehensive formula (also 'browlie')
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
        df["qb"] = 0.0
        motion = tau_shi >= tau_c_shi

        df.loc[motion, "qb"] = (
            8
            * (tau_shi[motion] - tau_c_shi) ** (1.5)
            * (info["d50"] * (g * (info["sg"] - 1) * info["d50"]) ** 0.5)
        )
        df["Qb"] = df["qb"] * 3600 * info["w"]

    elif type_ == "einstein-brown":

        w_s = settling_velocity(info["d50"], info)
        low_motion = tau_shi < 0.18
        mid_motion = (tau_shi >= 0.18) & (tau_shi < 0.52)
        high_motion = tau_shi >= 0.52
        df["qb"] = 0.0
        df.loc[low_motion, "qb"] = 2.15 * np.exp(-0.391 / tau_shi[low_motion])
        df.loc[mid_motion, "qb"] = 40 * tau_shi[mid_motion] ** 3
        df.loc[high_motion, "qb"] = 15 * tau_shi[high_motion] ** (1.5)

        df["qb"] = df["qb"] * w_s * info["d50"]
        df["Qb"] = df["qb"] * 3600 * info["w"]  # m3

    elif type_ == "wilcock-crowe":

        dm = np.sum(info["ds_substrate"] * info["frac_substrate"])
        Fs = info.get("Fs")  # Fraction of sand in the surface layer
        if Fs is None:
            Fs = 0.2
            logger.info("Percentage of sands if not given. Using Fs equals to 0.2.")

        tau_rm_shi = 0.021 + 0.015 * np.exp(-20 * Fs)
        tau_rm = tau_rm_shi * g * dm * info["rho_w"] * (info["sg"] - 1)  # reference shear stress (Pa)
        weight = np.zeros(len(tau))

        df["qb"] = 0.0

        for i in range(len(info["ds_substrate"])):
            b = 0.67 / (1 + np.exp(1.5 - info["ds_substrate"][i] / dm))
            tau_ri = tau_rm * (info["ds_substrate"][i] / dm) ** b
            Fi = tau / tau_ri
            mask = Fi < 1.35
            weight[mask] = 0.002 * Fi[mask] ** (7.5)
            weight[~mask] = 14 * (1 - 0.894 / ((Fi[~mask]) ** 0.5)) ** 4.5

            df["qb"] = df["qb"] + weight * info["frac_substrate"][i] * info["w"] * (
                g * df["h"] * info["S"]
            ) ** (3 / 2) * info["rho_s"] / (
                (info["sg"] - 1) * g
            )  # kg/s

        df["Qb"] = df["qb"] * 3600 / info["rho_s"]

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

        ds = np.atleast_1d(info["ds_substrate"])
        frac = np.atleast_1d(info.get("frac_substrate", np.full(len(ds), 1 / len(ds))))
        w_s = np.atleast_1d(settling_velocity(ds, info))

        h = df["h"].to_numpy(dtype=float)
        U = 1 / info["n"] * h ** (2 / 3) * info["S"] ** 0.5
        Q = U * info["w"] * h
        uc = (g * h * info["S"]) ** 0.5

        # Yang (1973) for sand and Yang (1984) for gravel; common logarithms,
        # concentration in ppm by weight
        qv = np.zeros(len(h))  # m3/s
        for d, ws, fr in zip(ds, w_s, frac):
            if d <= 0:
                continue
            re = uc * d / info["nu"]
            fac1 = np.zeros(len(h))  # critical velocity / fall velocity
            mask = (re > 1.2) & (re < 70)
            fac1[mask] = 2.5 / (np.log10(re[mask]) - 0.06) + 0.66
            fac1[re >= 70] = 2.05
            if d < 0.002:
                c = (5.435, 0.286, 0.457, 1.799, 0.409, 0.314)
            else:
                c = (6.681, 0.633, 4.816, 2.784, 0.305, 0.282)
            excess = U * info["S"] / ws - fac1 * info["S"]
            moving = excess > 0
            log_ct = (
                c[0]
                - c[1] * np.log10(ws * d / info["nu"])
                - c[2] * np.log10(uc[moving] / ws)
                + (
                    c[3]
                    - c[4] * np.log10(ws * d / info["nu"])
                    - c[5] * np.log10(uc[moving] / ws)
                )
                * np.log10(excess[moving])
            )
            qv[moving] += fr * 10**log_ct * 0.001 * Q[moving] / info["rho_s"]

        df["Qb"] = qv * 3600

    elif type_ in ("brownlie", "browlie"):
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
                raise ValueError(f"tau_adim '{info['tau_adim']}' not implemented.")
        else:
            logger.info("Browlie's tau adimensional parameter.")
            dens_apa = 1600
            Y_bw = (
                g**0.5
                * (dens_apa / info["rho_w"]) ** 0.5
                * info["d50"] ** (3 / 2)
                / info["nu"]
            ) ** (-0.6)
            tau_adim_c = 0.22 * Y_bw + 0.06 * 10 ** (-7.7 * Y_bw)

        sigma_g = info.get("sigma_g")  # geometric standard deviation of the bed material
        if sigma_g is None:
            sigma_g = 1.0
            logger.info("Geometric standard deviation sigma_g not given. Using 1 (uniform sediment).")
        cb = 1.268

        Uc = (
            ((info["sg"] - 1) * g * info["d50"]) ** 0.5
            * 4.596
            * tau_adim_c**0.529
            * info["S"] ** (-0.1405)
            * sigma_g ** (-0.1606)
        )
        df["Cppm"] = (
            7115
            * cb
            * (np.clip(U - Uc, 0, None) / ((info["sg"] - 1) * g * info["d50"]) ** 0.5) ** 1.978
            * info["S"] ** 0.6601
            * (df["h"] / info["d50"]) ** (-0.3301)
        )
        df["Qb"] = df["Cppm"] * 0.001 * Q * 3600 / info["rho_s"]

    return df["Qb"]
