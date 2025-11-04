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

__version__ = "2026.0.1"
__author__ = "Manuel Cobos"