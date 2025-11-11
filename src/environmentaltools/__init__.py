"""
Environmental Tools
===================

A comprehensive Python package for environmental data analysis and modeling.

This package provides tools for:
- Spatiotemporal analysis with BME methods
- Environmental indicator calculation
- Raster and vector data processing
- Statistical analysis and visualization
- Spectral data analysis
- Temporal series processing
- Drone mission planning and data generation

Modules:
- common: Core utilities and data I/O functions
- spatial: Geospatial analysis and topographic processing
- temporal: Time series analysis and statistical modeling
- spectral: Frequency domain analysis and tidal processing
- spatiotemporal: Combined spatial and temporal analysis with BME
- download: Environmental data acquisition from various sources
- drone: UAV mission planning and flight data generation
- graphics: Visualization and plotting utilities
- processes: Data processing workflows

Usage:
    Import specific modules as needed:
    
    >>> from environmentaltools import spatial, temporal
    >>> from environmentaltools.drone import calculate_scan_parameters
    >>> from environmentaltools.common import read

Logging Configuration
---------------------
The package uses loguru for clean, structured logging across all modules.
"""

import sys
from loguru import logger

# Configure loguru with a clean format for the entire package
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)