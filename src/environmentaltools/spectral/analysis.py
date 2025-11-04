"""
Spectral Analysis Module
=========================

This module provides comprehensive tools for spectral and harmonic analysis of time series data,
including:

- Lomb-Scargle periodogram analysis for unevenly sampled data
- Fast Fourier Transform (FFT) for regularly sampled data
- Harmonic (tidal) analysis using UTide
- Tidal reconstruction and prediction using pyTMD and EOT20 global tide model
- High-resolution tidal elevation predictions

The module supports both observational data analysis and model-based predictions for
oceanographic and environmental applications.
"""

# Standard library imports
import datetime
import importlib.util
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from urllib.request import urlretrieve

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyTMD.io
import pyTMD.predict
import pyTMD.tools
import pyTMD.utilities
import scipy.signal as scs
import timescale.time
import utide
from scipy.stats import chi2

# Local imports
from environmentaltools.common import utils, save, read
from loguru import logger


def lombscargle_periodogram(data, variable, max_period=None, nperiods=5, freq="H"):
    """
    Compute the Lomb-Scargle periodogram for unevenly sampled time series.
    
    The Lomb-Scargle periodogram is designed for spectral analysis of time series
    with irregular sampling. It identifies significant periodicities in the data
    by computing the power spectral density across a range of periods.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index.
    variable : str
        Name of the column containing the variable to analyze.
    max_period : float, optional
        Maximum period (in years for hourly data, or years for daily data) to include 
        in the analysis. If None, uses default range up to 2 years.
    nperiods : int, default=5
        Number of most significant periods to identify.
    freq : {'H', 'D'}, default='H'
        Frequency of the data: 'H' for hourly, 'D' for daily.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with periods as index and columns:
        
        - 'psd' : Power spectral density values
        - 'significant' : Boolean indicating the top n most significant periods
    
    Notes
    -----
    The function normalizes periods to years and applies a moving average with 
    window size of 100 to identify the most significant periodicities.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'temp': [20, 21, 19, 22]}, 
    ...                      index=pd.date_range('2020-01-01', periods=4, freq='H'))
    >>> psd = lombscargle_periodogram(data, 'temp', nperiods=2)
    """
    if freq == "D":
        time = (data.index - data.index[0]).days / 365.25
        periods = np.linspace(7 / 365.25, 2, 1000)
    else:
        time = (data.index - data.index[0]).total_seconds() / (365.25 * 24 * 3600)
        periods = np.linspace(24 * 7 / (24 * 365.25), 2, 1000)

    if max_period is not None:
        periods = np.hstack([periods, np.arange(2.01, max_period, 0.02)])

    freqs = 1.0 / periods
    angular_freq = 2 * np.pi * freqs

    psd = scs.lombscargle(
        time.values, data[variable].values, angular_freq, normalize=True
    )
    psd = pd.DataFrame(psd, index=periods, columns=["psd"])
    signf = utils.moving(psd, 100)
    # signf.columns, signf.index.name = ['PSD'], 'periods'
    signf = signf.nlargest(nperiods, "psd")
    psd["significant"] = False
    psd.loc[signf.index, "significant"] = True

    return psd


def fast_fourier_transform(data, variable, fname=None, freq="H", alpha=0.05):
    """
    Compute the Fast Fourier Transform for regularly sampled time series.
    
    This function performs spectral analysis using FFT, suitable for time series
    with regular sampling intervals. It identifies significant frequencies using
    a chi-squared test at the specified significance level.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index and regular sampling.
    variable : str
        Name of the column containing the variable to analyze.
    fname : str, optional
        Filename to save the power spectral density and frequencies. If None,
        results are not saved to file.
    freq : {'H', 'D'}, default='H'
        Frequency of the data: 'H' for hourly, 'D' for daily.
    alpha : float, default=0.05
        Significance level for the chi-squared test to identify significant frequencies.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with frequencies as index and columns:
        
        - 'psd' : Power spectral density values
        - 'significant' : Boolean indicating statistically significant frequencies
    
    Notes
    -----
    The spectral resolution (fmin) depends on the total duration of the time series.
    Longer time series provide finer frequency resolution. Only the positive frequency
    half of the spectrum is returned.
    
    The significance test uses a chi-squared distribution with degrees of freedom
    calculated based on the number of data points and the spectral resolution.
    """

    N = len(data)
    if freq == "H":
        # Spectral resolution: smaller values allow resolving more frequencies
        fmin = 1 / (
            (data.index[-1] - data.index[0]).total_seconds() / 3600
        )
    elif freq == "D":
        # Spectral resolution for daily data
        fmin = 1 / (
            (data.index[-1] - data.index[0]).days
        )

    coefs = np.fft.fft(data)  # - np.mean(data))

    # Choose one side of the spectra
    cn = np.ravel(coefs[0 : N // 2] / N)

    N_ = len(cn)
    f = np.arange(0, N_) * fmin
    S = 0.5 * np.abs(cn) ** 2 / fmin

    psd = pd.DataFrame(S, index=f, columns=["psd"])

    var = data.var().values
    M = N / 2
    phi = (2 * (N - 1) - M / 2.0) / M
    chi_val = chi2.isf(q=1 - alpha / 2, df=phi)  # /2 for two-sided test
    psd["significant"] = S > (var / N) * (chi_val / phi) / (S * f**2)

    return psd


def harmonic_analysis(data: pd.DataFrame, lat: float, file_name: str = None):
    """
    Perform harmonic (tidal) analysis on observational time series data.
    
    This function uses the UTide package (Codiga, 2011) to extract tidal constituents
    from time series data, including amplitude, phase, and their uncertainties. The
    method can handle time series with gaps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index. Missing values will be removed
        before analysis.
    lat : float
        Latitude of the observation location in decimal degrees. Required for
        computing nodal corrections.
    file_name : str, optional
        Path to save the tidal constituents as JSON file. If None, results are
        not saved to file.
    
    Returns
    -------
    dict
        Dictionary containing tidal constituents with keys:
        
        - Amplitude and phase for each constituent
        - Confidence intervals (from Monte Carlo analysis)
        - Statistical information about the fit
    
    Raises
    ------
    ValueError
        If the UTide library is not installed.
    
    Notes
    -----
    The analysis uses the ordinary least squares (OLS) method with nodal corrections
    and Monte Carlo confidence intervals. The Rayleigh criterion is set to 0.95
    to resolve closely spaced tidal constituents.
    
    References
    ----------
    Codiga, D.L., 2011. Unified Tidal Analysis and Prediction Using the UTide
    Matlab Functions. Technical Report 2011-01. Graduate School of Oceanography,
    University of Rhode Island, Narragansett, RI. 59pp.
    """

    lib_spec = importlib.util.find_spec("utide")
    if lib_spec is not None:
        from utide import solve
    else:
        raise ValueError(
            "You will require utide library. You can downloaded it from https://pypi.org/project/UTide/"
        )

    data.dropna(inplace=True)

    constituents = solve(
        data.index,
        data,
        lat=lat,
        nodal=True,
        trend=False,
        method="ols",
        conf_int="MC",
        Rayleigh_min=0.95,
        verbose=False,
    )

    if file_name is not None:
        save.to_json(constituents, file_name, True)

    return constituents


def reconstruct_tidal_level(df: pd.DataFrame, tidalConstituents: dict):
    """
    Reconstruct tidal elevations from harmonic constituents.
    
    This function uses tidal constituents (obtained from harmonic analysis) to
    reconstruct the tidal signal at specified times. The reconstruction includes
    all analyzed constituents with their phases and amplitudes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index specifying the times for reconstruction.
        A new column 'ma' will be added with the reconstructed tidal anomalies.
    tidalConstituents : dict
        Dictionary containing tidal constituents from harmonic analysis, typically
        the output from :func:`harmonic_analysis`. Must include
        amplitude, phase, and mean level information.
    
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'ma' column containing reconstructed tidal
        anomalies (reconstructed level minus mean level).
    
    Notes
    -----
    The 'ma' column represents tidal anomalies (deviations from mean level),
    which is useful for isolating tidal effects from other oceanographic processes.
    
    Uses the UTide reconstruct function to generate predictions from constituents.
    """
    from utide import reconstruct

    tidalLevel = reconstruct(df.index, tidalConstituents)

    df["ma"] = tidalLevel["h"] - tidalConstituents["mean"]
    return df


def check_and_download_tidal_model(model_name: str, database_path: str) -> tuple[bool, str]:
    """
    Check if tidal model files exist and download automatically if available.
    
    Currently supports automatic download for:
    - EOT20: Empirical Ocean Tide model (https://doi.org/10.17882/79489)
    - FES2014: Finite Element Solution tide model (requires AVISO credentials)
    - GOT4.10: Global Ocean Tide model
    
    Parameters
    ----------
    model_name : str
        Name of the tidal model (e.g., 'EOT20', 'FES2014', 'GOT4.10').
    database_path : str
        Path where model files should be stored.
        
    Returns
    -------
    tuple[bool, str]
        (success, format) where success is True if model files are available,
        and format is the recommended model format string.
        
    Notes
    -----
    - EOT20 can be downloaded automatically without credentials
    - FES2014 requires AVISO+ credentials (see pyTMD documentation)
    - TPXO models require registration and manual download
    """
    logger.info(f"Checking model files for {model_name} in {database_path}")
    
    # Create database directory if it doesn't exist
    db_path = Path(database_path)
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Define model-specific file patterns and download info
    model_info = {
        'EOT20': {
            'files': ['EOT20_load_tides.nc', 'EOT20_ocean_tides.nc'],
            'url_base': 'https://data.isimip.org/10.48364/ISIMIP.598515/',
            'downloadable': True,
            'format': 'netcdf'
        },
        'FES2014': {
            'files': ['ocean_tide/*nc'],
            'url': 'https://www.aviso.altimetry.fr/',
            'downloadable': False,
            'format': 'FES'
        },
        'GOT4.10': {
            'files': ['GOT4.10c*'],
            'downloadable': False,
            'format': 'GOT'
        },
        'TPXO9': {
            'files': ['TPXO9*'],
            'downloadable': False,
            'format': 'OTIS'
        }
    }
    
    # Check if model is known
    if model_name not in model_info:
        logger.warning(f"Model {model_name} not in automatic download list. Checking directory...")
        if not any(db_path.iterdir()):
            logger.error(f"No files found in {database_path}. Please download {model_name} manually.")
            return False, 'FES'
        return True, 'FES'
    
    info = model_info[model_name]
    
    # Check if model files already exist
    model_files_exist = False
    for pattern in info['files']:
        if '*' in pattern:
            if list(db_path.glob(pattern)):
                model_files_exist = True
                break
        else:
            if (db_path / pattern).exists():
                model_files_exist = True
                break
    
    if model_files_exist:
        logger.info(f"Model {model_name} files found in {database_path}")
        return True, info['format']
    
    # Try to download if supported
    if not info['downloadable']:
        logger.error(f"Model {model_name} requires manual download.")
        if 'url' in info:
            logger.error(f"Download from: {info['url']}")
        logger.error(f"Extract files to: {database_path}")
        return False, info['format']
    
    # Download EOT20
    if model_name == 'EOT20':
        logger.info(f"Downloading {model_name} model files...")
        try:
            for filename in info['files']:
                url = f"{info['url_base']}{filename}"
                output_file = db_path / filename
                logger.info(f"Downloading {filename}...")
                urlretrieve(url, output_file)
                logger.info(f"Downloaded {filename}")
            logger.info(f"{model_name} downloaded successfully")
            return True, info['format']
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            logger.info("You can download EOT20 manually from: https://doi.org/10.17882/79489")
            return False, info['format']
    
    return False, info['format']


def configure_tidal_model(
    model_name: str, 
    database_path: str, 
    model_format: str = None,
    auto_download: bool = True
) -> pyTMD.io.model:
    """
    Configure and initialize the tidal model with specified parameters.
    
    Parameters
    ----------
    model_name : str
        Name of the tidal model (e.g., 'EOT20', 'FES2014', 'TPXO9').
    database_path : str
        Path to the model database directory.
    model_format : str, optional
        Format of the model files (e.g., 'FES', 'netcdf', 'OTIS', 'ATLAS').
        If None, format is auto-detected based on model name.
    auto_download : bool, default=True
        If True, attempt to download model files automatically if not found.
        
    Returns
    -------
    pyTMD.io.model
        Configured tidal model instance
        
    Raises
    ------
    FileNotFoundError
        If model files are not found and cannot be downloaded
    ValueError
        If model configuration fails
        
    Notes
    -----
    The function will automatically download EOT20 model if files are missing
    and auto_download is True. Other models may require manual download.
    """
    logger.info(f"Configuring tidal model: {model_name}")
    
    # Create database path if it doesn't exist
    Path(database_path).mkdir(parents=True, exist_ok=True)
    
    # Check and download model files if necessary
    if auto_download:
        model_available, detected_format = check_and_download_tidal_model(model_name, database_path)
        if not model_available:
            raise FileNotFoundError(
                f"Model {model_name} files not found in {database_path} and could not be downloaded. "
                f"Please download manually."
            )
        # Use detected format if not explicitly provided
        if model_format is None:
            model_format = detected_format
            logger.info(f"Using auto-detected format: {model_format}")
    else:
        # Use default format if not provided
        if model_format is None:
            model_format = 'FES'
    
    # Configure model using modern pyTMD API
    model = pyTMD.io.model(
        database_path,
        format=model_format,
        compressed=True,
    ).elevation(model_name)
    
    logger.info(f"Model configured successfully with {len(model.constituents)} constituents")
    return model


def generate_datetime_series_from_range(
    start_date: datetime.date,
    end_date: datetime.date,
    resolution_minutes: int = 60
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Generate datetime series for tidal prediction using start and end dates.
    
    Parameters
    ----------
    start_date : datetime.date
        Starting date for the prediction
    end_date : datetime.date
        Ending date for the prediction
    resolution_minutes : int, optional
        Time resolution in minutes (default: 60)
        
    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex]
        tide_time: Time array in pyTMD format
        datetime_index: Pandas datetime index for CSV output
        
    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        logger.info(f"Generating datetime series from {start_date} to {end_date}")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        if resolution_minutes <= 0:
            raise ValueError("Time resolution must be positive")
        
        # Generate datetime index using pandas
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date
        
        datetime_index = pd.date_range(
            start=start_datetime,
            end=end_datetime,
            freq=f'{resolution_minutes}min',
            inclusive='left'  # Exclude the very last point to avoid going into next day
        )[:-1]  # Remove last point to stay within the range
        
        # Convert to pyTMD time format (days since some reference)
        tide_time = np.array([(dt - pd.Timestamp('1858-11-17')).total_seconds() / 86400.0 
                             for dt in datetime_index])
        
        logger.info(f"Generated {len(datetime_index)} time points")
        logger.info(f"Time range: {datetime_index[0]} to {datetime_index[-1]}")
        
        return tide_time, datetime_index
        
    except Exception as e:
        logger.error(f"Failed to generate datetime series: {str(e)}")
        raise ValueError(f"Datetime series generation failed: {str(e)}")


def generate_datetime_series(
    start_date: datetime.date,
    forecast_days: int,
    resolution_minutes: int = 60
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Generate datetime series for tidal prediction with proper datetime indexing.
    
    Parameters
    ----------
    start_date : datetime.date
        Starting date for the prediction
    forecast_days : int
        Number of days to forecast
    resolution_minutes : int, optional
        Time resolution in minutes (default: 60)
        
    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex]
        tide_time: Time array in pyTMD format
        datetime_index: Pandas datetime index for CSV output
        
    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    logger.info(f"Generating datetime series from {start_date} for {forecast_days} days")
    
    if forecast_days <= 0:
        raise ValueError("Forecast days must be positive")
    if resolution_minutes <= 0:
        raise ValueError("Time resolution must be positive")
    
    # Generate datetime index using pandas
    start_datetime = pd.Timestamp(start_date)
    end_datetime = start_datetime + pd.Timedelta(days=forecast_days)
    
    datetime_index = pd.date_range(
        start=start_datetime,
        end=end_datetime,
        freq=f'{resolution_minutes}min',
        inclusive='left'  # Exclude the end point
    )
    
    # Calculate minutes array for pyTMD conversion
    total_minutes = len(datetime_index) * resolution_minutes
    minutes = np.arange(0, total_minutes, resolution_minutes)[:len(datetime_index)]
    
    # Convert time from calendar date to pyTMD format
    # In modern pyTMD, time conversion is more straightforward
    # Modern timescale.time API
    tide_time = timescale.time.convert_calendar_dates(
        start_date.year, start_date.month, start_date.day, minute=minutes
    )
    logger.info("Using modern timescale.time.convert_calendar_dates")
    
    logger.info(f"Generated {len(tide_time)} time points")
    logger.info(f"Time range: {datetime_index[0]} to {datetime_index[-1]}")
    
    return tide_time, datetime_index


def save_results_to_csv(
    tidal_data: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    output_directory: str,
    location_name: str,
    obs_data: Optional[np.ndarray] = None,
    decimal_places: int = 3
) -> str:
    """
    Save tidal prediction results to CSV file with proper datetime formatting.
    
    Parameters
    ----------
    tidal_data : np.ndarray
        Predicted tidal elevations
    datetime_index : pd.DatetimeIndex
        Datetime index for the data
    output_directory : str
        Directory where the CSV file will be saved
    location_name : str
        Name of the location for the filename
    obs_data : np.ndarray, optional
        Observational data for comparison (if available)
    decimal_places : int, default=3
        Number of decimal places for elevation values
        
    Returns
    -------
    str
        Path to the saved CSV file
        
    Raises
    ------
    ValueError
        If saving fails
    """
    logger.info("Saving results to CSV file")
    
    # Create DataFrame
    data_dict = {
        'datetime': datetime_index,
        'tidal_elevation_m': np.round(tidal_data, decimal_places)
    }
    
    # Add observational data if available
    if obs_data is not None and len(obs_data) == len(tidal_data):
        data_dict['observed_m'] = np.round(obs_data, decimal_places)
        data_dict['residual_m'] = np.round(obs_data - tidal_data, decimal_places)
    
    df = pd.DataFrame(data_dict)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tidal_predictions_{location_name.replace(' ', '_')}_{timestamp}.csv"
    filepath = Path(output_directory) / filename
    
    # Save to CSV
    df.to_csv(
        filepath,
        index=False,
        sep=',',
        date_format='%Y-%m-%d %H:%M:%S',
        float_format=f'%.{decimal_places}f'
    )
            
    return str(filepath)



def generate_time_series(
    start_date: datetime.date,
    forecast_days: int,
    resolution_minutes: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time series for tidal prediction.
    
    Parameters
    ----------
    start_date : datetime.date
        Starting date for the prediction
    forecast_days : int
        Number of days to forecast
    resolution_minutes : int, optional
        Time resolution in minutes (default: 1)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tide_time: Time array in pyTMD format
        hours: Time array in hours
        
    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    logger.info(f"Generating time series from {start_date} for {forecast_days} days")
    
    if forecast_days <= 0:
        raise ValueError("Forecast days must be positive")
    if resolution_minutes <= 0:
        raise ValueError("Time resolution must be positive")
    
    # Calculate total minutes for the forecast period
    total_minutes = forecast_days * 24 * 60
    minutes = np.arange(0, total_minutes, resolution_minutes)
    
    # Convert time from calendar date to pyTMD format
    # In modern pyTMD, time conversion is more straightforward
    # Modern timescale.time API
    tide_time = timescale.time.convert_calendar_dates(
        start_date.year, start_date.month, start_date.day, minute=minutes
    )
    logger.info("Using modern timescale.time.convert_calendar_dates")    
    hours = minutes / 60.0
    
    logger.info(f"Generated {len(tide_time)} time points")
    return tide_time, hours



def interpolate_tidal_constants(
    longitude: float,
    latitude: float,
    model: pyTMD.io.model,
    method: str = "spline",
    extrapolate: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate tidal harmonic constants for specified location.
    
    Parameters
    ----------
    longitude : float
        Longitude of the location (degrees East)
    latitude : float
        Latitude of the location (degrees North)
    model : pyTMD.io.model
        Configured tidal model instance
    method : str, optional
        Interpolation method (default: "spline")
    extrapolate : bool, optional
        Whether to extrapolate beyond model domain (default: True)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        amplitude: Tidal constituent amplitudes
        phase: Tidal constituent phases in degrees
        
    Raises
    ------
    ValueError
        If interpolation fails or coordinates are invalid
    """
    logger.info(f"Interpolating tidal constants for location: ({latitude:.6f}°N, {longitude:.6f}°E)")
    
    # Validate coordinates
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Invalid latitude: {latitude}")
    if not (-180 <= longitude <= 360):
        raise ValueError(f"Invalid longitude: {longitude}")
    
    # Read tidal constants from model
    constituents = pyTMD.io.FES.read_constants(
        model.model_file,
        type=model.type,
        version=model.version,
        compressed=model.compressed,
    )
    
    # Interpolate constants to location
    amplitude, phase = pyTMD.io.FES.interpolate_constants(
        np.atleast_1d(longitude),
        np.atleast_1d(latitude),
        constituents,
        scale=model.scale,
        method=method,
        extrapolate=extrapolate,
    )
    
    logger.info(f"Interpolated {len(amplitude)} harmonic constants")
    return amplitude, phase


def predict_tidal_elevations(
    tide_time: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    constituents: list,
    model: pyTMD.io.model,
    include_minor: bool = True,
    unit_factor: float = 100.0
) -> np.ndarray:
    """
    Predict tidal elevations using harmonic analysis.
    
    Parameters
    ----------
    tide_time : np.ndarray
        Time array in pyTMD format
    amplitude : np.ndarray
        Tidal constituent amplitudes
    phase : np.ndarray
        Tidal constituent phases in degrees
    constituents : list
        List of tidal constituents
    model : pyTMD.io.model
        Configured tidal model instance
    include_minor : bool, optional
        Whether to include minor constituent inference (default: True)
    unit_factor : float, optional
        Unit conversion factor (default: 100.0 for cm)
        
    Returns
    -------
    np.ndarray
        Predicted tidal elevations in specified units
        
    Raises
    ------
    ValueError
        If prediction calculation fails
    """
    logger.info("Computing tidal predictions")
    
    # In modern pyTMD versions, delta time corrections are handled automatically
    # or through different methods. Let's try the modern approach first.
    
    # Calculate complex harmonic constants
    # Convert phase to radians and compute complex representation
    complex_phase = -1j * phase * np.pi / 180.0
    harmonic_constants = amplitude * np.exp(complex_phase)
    
    # Try modern pyTMD prediction without explicit deltat
    try:
        # Modern API - pyTMD handles time corrections internally
        tide_prediction = pyTMD.predict.time_series(
            tide_time, 
            harmonic_constants, 
            constituents, 
            corrections=model.format
        )
        logger.info("Using modern pyTMD API without explicit deltat")
        
    except Exception as e1:
        logger.debug(f"Modern API without deltat failed: {e1}")
        
        # Fallback: Try with deltat=None (some versions accept this)
        try:
            tide_prediction = pyTMD.predict.time_series(
                tide_time, 
                harmonic_constants, 
                constituents, 
                deltat=None,
                corrections=model.format
            )
            logger.info("Using pyTMD API with deltat=None")
            
        except Exception as e2:
            logger.debug(f"API with deltat=None failed: {e2}")
            
            # Last resort: Try with zero deltat array
            try:
                deltat = np.zeros_like(tide_time)
                tide_prediction = pyTMD.predict.time_series(
                    tide_time, 
                    harmonic_constants, 
                    constituents, 
                    deltat=deltat,
                    corrections=model.format
                )
                logger.info("Using pyTMD API with zero deltat array")
                
            except Exception as e3:
                # Try without corrections parameter
                try:
                    tide_prediction = pyTMD.predict.time_series(
                        tide_time, 
                        harmonic_constants, 
                        constituents
                    )
                    logger.info("Using simplified pyTMD API without corrections")
                except Exception as e4:
                    raise ValueError(f"All prediction methods failed. Last error: {str(e4)}")
    
    # Add minor constituent inference if requested
    if include_minor:
        logger.info("Including minor constituent inference")
        try:
            # Try modern minor constituent inference
            minor_constituents = pyTMD.predict.infer_minor(
                tide_time, 
                harmonic_constants, 
                constituents, 
                corrections=model.format
            )
        except Exception:
            # Fallback for minor constituents
            try:
                minor_constituents = pyTMD.predict.infer_minor(
                    tide_time, 
                    harmonic_constants, 
                    constituents
                )
            except Exception as e_minor:
                logger.warning(f"Minor constituent inference failed: {str(e_minor)}")
                logger.info("Proceeding without minor constituents")
                minor_constituents = None
        
        if minor_constituents is not None:
            tide_prediction.data[:] += minor_constituents.data[:]
    
    # Apply unit conversion
    tidal_elevations = tide_prediction.data * unit_factor
    
    logger.info(f"Prediction completed: {len(tidal_elevations)} points")
    return tidal_elevations


def tidal_reconstruction_from_models(
    latitude: float,
    longitude: float,
    start_date: datetime.date,
    end_date: datetime.date,
    model_name: str = 'EOT20',
    database_path: str = None,
    time_resolution_minutes: int = 60,
    output_directory: str = '.',
    location_name: str = 'Unknown'
) -> Dict[str, Any]:
    """
    Tidal reconstruction and prediction from global tide models.
    
    This function provides a streamlined interface for tidal predictions using
    global tide models (e.g., EOT20, FES2014, TPXO9). 
    It performs the complete workflow: model configuration,
    interpolation of tidal constants, prediction of elevations, and CSV export.
    
    Parameters
    ----------
    latitude : float
        Latitude of the location in decimal degrees (-90 to 90).
    longitude : float
        Longitude of the location in decimal degrees (-180 to 360).
    start_date : datetime.date
        Starting date for the prediction period.
    end_date : datetime.date
        Ending date for the prediction period.
    model_name : str, default='EOT20'
        Name of the tidal model to use (e.g., 'EOT20', 'FES2014', 'TPXO9').
    database_path : str, optional
        Path to the tidal model database directory. If None, uses current directory.
    time_resolution_minutes : int, default=60
        Time resolution for predictions in minutes (e.g., 60 for hourly).
    output_directory : str, default='.'
        Directory where output files (CSV and plots) will be saved.
    location_name : str, default='Unknown'
        Name of the location for labeling outputs.
    
    Returns
    -------
    dict
        Dictionary containing:
        
        - 'predictions' : np.ndarray - Tidal elevation predictions
        - 'datetime_index' : pd.DatetimeIndex - Time stamps for predictions
        - 'location' : tuple - (latitude, longitude)
        - 'csv_file' : str - Path to saved CSV file
        - 'plot_file' : str - Path to saved plot
    
    Examples
    --------
    >>> import datetime
    >>> results = tidal_reconstruction_from_models(
    ...     latitude=41.173,
    ...     longitude=-8.720,
    ...     start_date=datetime.date(2024, 1, 1),
    ...     end_date=datetime.date(2024, 1, 31),
    ...     location_name='Porto'
    ... )
    
    Notes
    -----
    This function is a simplified alternative to using complex configuration.
    For harmonic analysis from observational data, use :func:`harmonic_analysis` 
    and :func:`reconstruct_tidal_level`.
    """
    logger.info(f"Location: {location_name} ({latitude:.6f}°N, {longitude:.6f}°E)")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Resolution: {time_resolution_minutes} minutes")
    
    # Set default database path if not provided
    if database_path is None:
        database_path = '.'

    
    
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Configure tidal model (using default FES format)
    logger.info(f"Configuring model: {model_name}")
    model = configure_tidal_model(model_name, database_path)
    
    # Generate prediction time series
    logger.info("Generating time series")
    tide_time, datetime_index = generate_datetime_series_from_range(
        start_date,
        end_date,
        time_resolution_minutes
    )
    
    # Interpolate tidal constants for location
    logger.info("Interpolating tidal constants")
    amplitude, phase = interpolate_tidal_constants(
        longitude, latitude, model,
        method='spline',
        extrapolate=False
    )
    
    # Predict tidal elevations
    logger.info("Predicting tidal elevations")
    tidal_predictions = predict_tidal_elevations(
        tide_time, amplitude, phase, model.constituents, model,
        include_minor=True, unit_factor=1.0
    )
    
    # Save results to CSV
    logger.info("Saving results to CSV")
    csv_filepath = save_results_to_csv(
        tidal_predictions, 
        datetime_index, 
        output_directory, 
        location_name
    )

    return {
        'predictions': tidal_predictions,
        'datetime_index': datetime_index,
        'location': (latitude, longitude),
        'csv_file': csv_filepath
    }
    