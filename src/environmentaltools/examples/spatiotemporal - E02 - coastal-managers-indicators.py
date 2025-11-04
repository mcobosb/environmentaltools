"""
Coastal Management Indicators Example

This script demonstrates how to use the spatiotemporal raster module to compute
coastal management indicators from DTM (Digital Terrain Model) data.

The script reads configuration from settings.ini and config.json files to set up
processing parameters for coastal flood risk analysis.
"""

import os
from pathlib import Path
from environmentaltools.spatiotemporal.raster import analysis
from loguru import logger

logger.info("="*60)
logger.info("COASTAL MANAGEMENT INDICATORS EXAMPLE")
logger.info("="*60)

cwd = os.path.dirname(os.path.abspath(__file__))
config_path = Path("C:/Users/m_cob/Desktop/test_raster_tool/coastal_management/config.json")

logger.info(f"Using configuration file: {config_path}")
logger.info("Starting analysis...")

# Execute the raster analysis pipeline
analysis(config_path)
