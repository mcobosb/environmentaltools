"""Physical properties of sea water and sediment: density, viscosity and fall velocity."""

import numpy as np


def fall_velocity(d, T, S):
    """Estimate sediment particle fall velocity using Soulsby's (1997) formula.

    Calculates settling velocity accounting for water temperature and salinity effects
    on density and viscosity.

    Parameters
    ----------
    d : float or np.ndarray
        Grain diameter (mm)
    T : float
        Water temperature (°C)
    S : float
        Salinity (ppt or ‰)

    Returns
    -------
    w : float or np.ndarray
        Fall velocity (m/s)

    Notes
    -----
    Uses Soulsby (1997) optimization:
        w = ν/d * [sqrt(10.36² + 1.049 * D³) - 10.36]

    where:
    - ν is kinematic viscosity (m²/s)
    - D = [(g(s-1)/ν²)]^(1/3) * d is dimensionless grain size
    - s = ρₛ/ρ is specific gravity

    Valid for natural sediments across wide size range (0.1 μm to 10 mm).
    Accounts for temperature/salinity effects through density and viscosity.

    References
    ----------
    Soulsby, R. (1997). Dynamics of marine sands. Thomas Telford.
    """

    g = 9.81
    rho = density(T, S)
    kvis = kinematic_viscosity(T)
    rhos = 2650
    d = d / 1000
    s = rhos / rho
    D = (g * (s - 1) / kvis**2) ** (1 / 3) * d
    w = kvis / d * (np.sqrt(10.36**2 + 1.049 * D**3) - 10.36)
    return w


def density(T, S):
    """Estimate water density from temperature and salinity.

    Uses empirical approximation for seawater density as function of temperature
    and salinity.

    Parameters
    ----------
    T : float
        Water temperature (°C)
    S : float
        Salinity (‰ or ppt)

    Returns
    -------
    rho : float
        Water density (kg/m³)

    Notes
    -----
    Approximation from Van Rijn (1993):
        ρ = 1000 + 1.455 * CL - 6.5×10⁻³ * (T - 4 + 0.4*CL)²

    where CL = (S - 0.03) / 1.805 is chlorinity.

    Valid for typical oceanographic conditions (S = 0-40‰, T = 0-30°C).

    References
    ----------
    Van Rijn, L. C. (1993). Principles of sediment transport in rivers, estuaries
    and coastal seas. Aqua Publications.
    """

    CL = (S - 0.03) / 1.805  # VanRijn

    rho = 1000 + 1.455 * CL - 6.5e-3 * (T - 4 + 0.4 * CL) ** 2
    return rho


def kinematic_viscosity(T):
    """Estimate kinematic viscosity of water from temperature.

    Calculates temperature-dependent kinematic viscosity using empirical formula.

    Parameters
    ----------
    T : float
        Water temperature (°C)

    Returns
    -------
    kvis : float
        Kinematic viscosity (m²/s)

    Notes
    -----
    Approximation from Van Rijn (1989).
    Kinematic viscosity decreases with increasing temperature.
    
    Typical values:
    - At 0°C: ν ≈ 1.79 × 10⁻⁶ m²/s
    - At 10°C: ν ≈ 1.31 × 10⁻⁶ m²/s
    - At 20°C: ν ≈ 1.00 × 10⁻⁶ m²/s

    References
    ----------
    Van Rijn, L. C. (1989). Handbook of Sediment Transport by Currents and Waves.
    """

    kvis = 1e-6 * (1.14 - 0.031 * (T - 15) + 6.8e-4 * (T - 15) ** 2)
    return kvis


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
        d_ast = ds * ((info["sg"] - 1) * g / info["nu"] ** 2) ** (1 / 3)  # dimensionless diameter
        w_s = (
            M
            * info["nu"]
            / (N * ds)
            * (np.sqrt(0.25 + (4 * N / (3 * M**2) * d_ast**3) ** (1 / n)) - 0.5) ** n
        )

    else:
        raise ValueError("Settling velocity formula not implemented.")

    return w_s


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
