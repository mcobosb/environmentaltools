import numpy as np
from scipy.optimize import fsolve


def frequency_limits(Te):
    """Obtain cutoff frequencies for Baquerizo's wave separation method.

    Calculates the frequency range for spectral analysis of incident and reflected
    waves, centered around the peak frequency.

    Parameters
    ----------
    Te : float
        Wave energy period (s)

    Returns
    -------
    f1 : float
        Lower cutoff frequency (Hz), equal to 0.5 * peak frequency
    f2 : float
        Upper cutoff frequency (Hz), equal to 1.5 * peak frequency
    fp : float
        Peak frequency (Hz), equal to 1/Te

    Notes
    -----
    All frequencies are rounded to 2 decimal places.
    This defines the spectral window for wave reflection analysis.
    """

    fp = 1 / Te
    f1, f2 = 0.5 * fp, 1.5 * fp

    fp, f1, f2 = np.round(fp, 2), np.round(f1, 2), np.round(f2, 2)
    return f1, f2, fp


def clsquare_s(eta, x, h, dt, eps, f1, f2):
    """Calculate complex amplitudes of incident and reflected wave trains.

    Implements Baquerizo's (1995) least squares method to separate incident and
    reflected wave components from time series measured by three aligned sensors.

    Parameters
    ----------
    eta : np.ndarray
        Free surface elevation (m), shape (ndat, 3) for three sensors
    x : np.ndarray
        Sensor positions along propagation direction (m), length 3
    h : float
        Water depth (m)
    dt : float
        Sampling interval (s)
    eps : float
        Tolerance, minimum denominator value to avoid singularities
    f1 : float
        Minimum analysis frequency (Hz)
    f2 : float
        Maximum analysis frequency (Hz)

    Returns
    -------
    f : np.ndarray
        Frequencies within [f1, f2] range (Hz)
    Zi : np.ndarray
        Complex amplitudes of incident wave train (m)
    Zr : np.ndarray
        Complex amplitudes of reflected wave train (m)

    References
    ----------
    Baquerizo, A. (1995). Reflexión del oleaje en playas. Método de las tres
    sondas. Revista de Obras Públicas.

    Notes
    -----
    The method uses Fourier analysis and the dispersion relation to separate
    waves propagating in opposite directions. Requires three aligned wave gauges.
    """

    c = np.conj(np.fft.fft(eta.T)).T
    ndat = len(eta)
    df = 1 / (ndat * dt)

    k, f = np.zeros(np.int(ndat / 2.0 + 1)), np.zeros(np.int(ndat / 2.0 + 1))
    S_mas = np.zeros([len(k)], dtype=np.complex)
    S_min = np.zeros([len(k)], dtype=np.complex)
    Bi = np.zeros([len(k)], dtype=np.complex)
    Br = np.zeros([len(k)], dtype=np.complex)

    for j in range(1, ndat / 2 + 1):
        f[j] = j * df
        k[j] = wave_number(1 / f[j], h)
        S_mas[j] = np.sum(np.exp(2.0 * 1j * k[j] * x))
        S_min[j] = np.sum(np.exp(-2.0 * 1j * k[j] * x))

        Bi[j] = np.sum(c[j, :] * np.exp(-1j * k[j] * x))
        Br[j] = np.sum(c[j, :] * np.exp(1j * k[j] * x))

    mask = (f > f1) & (f < f2) & (np.abs(9.0 - S_mas * S_min) > eps)

    Zi = np.conj(
        (3.0 * Br[mask] - Bi[mask] * S_mas[mask]) / (9.0 - S_mas[mask] * S_min[mask])
    )
    Zr = np.conj(
        (3.0 * Bi[mask] - Br[mask] * S_min[mask]) / (9.0 - S_mas[mask] * S_min[mask])
    )

    Zi, Zr = Zi / ndat, Zr / ndat
    return f[mask], Zi, Zr


def wave_number(t, h):
    """Calculate wave number from dispersion relation.

    Solves the linear wave dispersion relation to compute wave number k given
    wave period T and water depth h. Uses iterative solver with appropriate
    initial guesses for deep and shallow water.

    Parameters
    ----------
    t : float, int, or np.ndarray
        Wave period (s). Can be single value or array.
    h : float
        Water depth (m)

    Returns
    -------
    k : np.ndarray
        Wave number (rad/m)

    Notes
    -----
    Dispersion relation: σ² = g * k * tanh(k * h)
    where σ = 2π/T is the angular frequency and g = 9.81 m/s².

    Initial guess selection:
    - Deep water (γ > π²): k₀ = σ²/g
    - Shallow water (γ ≤ π²): k₀ = σ²/(gh)
    where γ = σ² * h / g

    Uses scipy.optimize.fsolve for iterative solution.
    """

    if isinstance(t, (int, float)):
        t = np.array([t])

    gamma = (2 * np.pi / t) ** 2.0 * h / 9.81
    sigma2 = (2 * np.pi / t) ** 2

    def func(k, sigma2, h):
        g = 9.81
        return sigma2 / g - k * np.tanh(k * h)

    k = np.zeros(len(t))
    for ind_, period in enumerate(t):
        if gamma[ind_] > np.pi**2:
            ko = np.sqrt(gamma[ind_] / h)
            k[ind_] = fsolve(func, ko, args=(sigma2[ind_], h))
        else:
            ko = gamma[ind_] / h
            k[ind_] = fsolve(func, ko, args=(sigma2[ind_], h))

        k[ind_] = np.abs(k[ind_])
    return k


def calculate_wave_reflection(T, mdat, t, xsn, h, dt):
    """Calculate wave reflection parameters from multi-sensor measurements.

    Analyzes time series from three aligned wave sensors to compute incident
    wave height, reflection coefficient, and phase difference using spectral
    separation methods.

    Parameters
    ----------
    T : float
        Wave energy period (s)
    mdat : np.ndarray
        Surface elevation measurements (m), shape (ndat, 3) for three sensors
    t : np.ndarray
        Time vector (s), currently unused but kept for compatibility
    xsn : np.ndarray
        Sensor positions along wave propagation axis (m), length 3
    h : float
        Water depth (m)
    dt : float
        Sampling interval (s)

    Returns
    -------
    HI : float
        Incident wave height (m)
    R : float
        Reflection coefficient (dimensionless, 0-1)
    fi : float
        Phase difference between incident and reflected waves (radians)
    f : np.ndarray
        Frequency vector within analysis range (Hz)
    Zi : np.ndarray
        Complex amplitudes of incident waves (m)
    Zr : np.ndarray
        Complex amplitudes of reflected waves (m)

    Notes
    -----
    The reflection coefficient R is defined as:
    R = sqrt(Σ|Zr|² / Σ|Zi|²)

    Incident wave height is computed from the zeroth moment of the incident spectrum:
    HI = sqrt(8 * m₀)

    Uses Baquerizo's least squares method for wave separation.
    """

    eps = 1e-10
    [f1, f2, fp] = frequency_limits(T)

    eta = mdat[:, :]

    # 2 - Separacion incidente-reflejada
    [f, Zi, Zr] = clsquare_s(eta, xsn, h, dt, eps, f1, f2)

    # 3 - R y fi
    ZiR = np.abs(2.0 * Zi) ** 2.0
    ZrR = np.abs(2.0 * Zr) ** 2.0

    # Espectro incidente
    df = f[1] - f[0]
    Si = ZiR / (2.0 * df)

    # Momento de primer orden
    moI = df * (0.5 * Si[0] + 0.5 * Si[-1] + np.sum(Si[1:-2]))
    # Altura de ola incidente
    HI = np.sqrt(8 * moI)

    # Coeficiente de reflexion
    R = np.sqrt(np.sum(ZrR) / np.sum(ZiR))
    index = np.argmax(Zi)
    # Fase de la reflexión
    fi = np.angle(Zi[index]) - np.angle(Zr[index])

    return HI, R, fi, f, Zi, Zr


def closure_depth(data, Hallermeier=True):
    """Compute closure depth using Hallermeier or Birkemeier empirical formula.

    Calculates the depth beyond which significant seabed changes are unlikely to occur
    over an annual cycle. Uses wave statistics exceeded 12 hours per year.

    Parameters
    ----------
    data : pd.DataFrame
        Wave characteristics time series with columns:
        - Hm0 : Significant wave height (m)
        - Tp : Peak wave period (s)
        Index should be datetime format
    Hallermeier : bool, optional
        If True, use Hallermeier (1981) formula; if False, use Birkemeier (1985).
        Default: True

    Returns
    -------
    dc : float
        Closure depth (m)

    Notes
    -----
    Hallermeier (1981) formula:
        dc = 2.28 * H₁₂ - 68.5 * (H₁₂² / (g * T₁₂²))

    Birkemeier (1985) formula:
        dc = 1.75 * H₁₂ - 57.9 * (H₁₂² / (g * T₁₂²))

    where H₁₂ and T₁₂ are wave height and period exceeded 12 hours per year.

    The closure depth separates the active nearshore zone from the inactive offshore zone.

    References
    ----------
    - Hallermeier, R. J. (1981). A profile zonation for seasonal sand beaches
      from wave climate. Coastal Engineering, 4, 253-277.
    - Birkemeier, W. A. (1985). Field data on seaward limit of profile change.
      Journal of Waterway, Port, Coastal, and Ocean Engineering, 111(3), 598-602.
    """
    G = 9.81
    list_H12, list_T12 = [], []
    for year in data.index.year.unique():
        # For every year, obtain the location of 12 hours exceedance
        data_year = data.loc[str(year), :].dropna()  # removing nans
        isort = np.argsort(data_year["Hm0"]).values  # index of 12 hrs

        # to ensure there is enough hourly data in a year (24*365.25 = 8766)
        if len(data_year) < 8766 / 2:
            list_H12.append(data_year.loc[str(year), "Hm0"].values[isort[-12]])
            list_T12.append(data_year.loc[str(year), "Tp"].values[isort[-12]])

    # Compute the average for all years
    H12, T12 = np.mean(np.asarray(list_H12)), np.mean(np.asarray(list_T12))
    if Hallermeier:
        dc = 2.28 * H12 - 68.5 * (H12**2 / (G * T12**2))
    else:
        dc = 1.75 * H12 - 57.9 * (H12**2 / (G * T12**2))
    return dc




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


def zero_cross(ts, dt):
    """Analyze time series using zero-upcrossing method.

    Identifies individual waves in a time series by detecting zero-upcrossing points
    and computing wave-by-wave statistics (heights and periods).

    Parameters
    ----------
    ts : np.ndarray
        Free surface elevation time series (m)
    dt : float
        Temporal resolution, sampling interval (s), dt = 1/fs where fs is sampling frequency

    Returns:
        Una tupla de la forma:
        * ``H``: vector con la serie de alturas de ola [L]
        * ``T``: vector con la serie de periodos de paso por cero [T]
        * ``Ac``: vector con las amplitudes de cresta [L]
        * ``As``: vector con las amplitudes de seno [L]
        * ``Tc``: vector con los periodos de cresta [T]
        * ``Ts``: vector con los periodos de seno [T]

    """

    ndat = len(ts)
    t = np.linspace(0, ndat - 1, ndat - 1) * dt

    # Cálculo del nº de pasos ascendentes por cero.
    sg = np.sign(ts)
    ps = sg[0 : ndat - 1] * sg[1:ndat]

    i1 = np.asarray([i for i, x in enumerate(ps) if x < 0])
    i2 = np.zeros(len(i1), dtype=int)

    vmx = np.zeros(len(i1) - 1)
    imx = np.zeros(len(i1) - 1, dtype=int)
    for j in range(0, len(i1) - 2):
        vmx[j] = max(abs(ts[i1[j] + 1 : i1[j + 1] + 1]))
        io = [
            float(item)
            for item, jj in enumerate(abs(ts[i1[j] : i1[j + 2]]))
            if jj == vmx[j]
        ]
        i2[j] = int(i1[j] + 1)
        imx[j] = int(i1[j] + io[0])
    imx[-1] = i1[-1]

    tc = t[i1] - dt * ts[i1] / (ts[i2] - ts[i1])
    dtc = np.diff(tc)

    if ts[0] < 0:  # El primero es ascendente
        Tc = dtc[0::2]
        Ts = dtc[1::2]
        ic = imx[0::2]
        it = imx[1::2]
        Ac = vmx[0::2]
        As = vmx[1::2]
        if ts[-1] < 0:  # El último es ascendente
            Tc, Ac, ic = Tc[0:-1], Ac[0:-1], ic[0:-1]
            T = Tc + Ts
            H = Ac + As
            i1 = i1[0:-1]
        else:
            T = Tc + Ts
            H = Ac + As

    else:  # El primero es descendente
        Tc = dtc[1::2]
        Ts = dtc[2::2]
        ic = imx[1::2]
        it = imx[2::2]
        Ac = vmx[1::2]
        As = vmx[2::2]
        if ts[-1] < 0:  # El último es ascendente
            Tc, Ac, ic = Tc[0:-1], Ac[0:-1], ic[0:-1]
            T = Tc + Ts
            H = Ac + As
            i1 = i1[1:-1]
        else:
            T = Tc + Ts
            H = Ac + As
            i1 = i1[1:]

    return H, T, Ac, As, Tc, Ts, ic, it, i1