"""
Spectral Analysis Module
=========================

This module provides comprehensive tools for spectral and harmonic analysis of 
environmental time series data.

Main Components
---------------

**Spectral Analysis**
    - Lomb-Scargle periodogram for unevenly sampled data
    - Fast Fourier Transform (FFT) for regularly sampled data

**Harmonic (Tidal) Analysis from Observations**
    - UTide-based harmonic analysis to extract tidal constituents from observations
    - Tidal level reconstruction from constituents

**Tidal Predictions from Global Models**
    - Configuration and interpolation from global tide models (EOT20, FES2014, TPXO9)
    - High-resolution tidal elevation predictions
    - Simplified workflow function for model-based predictions

Examples
--------
Spectral analysis:
    >>> psd = lombscargle_periodogram(data, 'variable', nperiods=5)
    
Harmonic analysis from observations:
    >>> constituents = harmonic_analysis(obs_data, lat=41.173)
    >>> reconstructed = reconstruct_tidal_level(df, constituents)
    
Tidal predictions from models:
    >>> import datetime
    >>> results = tidal_reconstruction_from_models(
    ...     latitude=41.173, longitude=-8.720,
    ...     start_date=datetime.date(2024, 1, 1),
    ...     end_date=datetime.date(2024, 1, 31)
    ... )
"""

from environmentaltools.spectral.analysis import (
    # Spectral analysis functions
    lombscargle_periodogram,
    fast_fourier_transform,
    
    # Harmonic (tidal) analysis from observations
    harmonic_analysis,
    reconstruct_tidal_level,
    
    # Tidal model configuration and utilities
    check_and_download_tidal_model,
    configure_tidal_model,
    generate_datetime_series_from_range,
    generate_datetime_series,
    generate_time_series,
    
    # Tidal predictions from models
    interpolate_tidal_constants,
    predict_tidal_elevations,
    
    # Output and main workflow
    save_results_to_csv,
    tidal_reconstruction_from_models,
)

__all__ = [
    # Spectral analysis
    'lombscargle_periodogram',
    'fast_fourier_transform',
    
    # Harmonic analysis from observations
    'harmonic_analysis',
    'reconstruct_tidal_level',
    
    # Tidal model configuration
    'check_and_download_tidal_model',
    'configure_tidal_model',
    
    # Time series utilities
    'generate_datetime_series_from_range',
    'generate_datetime_series',
    'generate_time_series',
    
    # Tidal predictions from models
    'interpolate_tidal_constants',
    'predict_tidal_elevations',
    
    # Output and workflow
    'save_results_to_csv',
    'tidal_reconstruction_from_models',
]

