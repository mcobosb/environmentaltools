"""
Coastal Management Indicators Example

This script demonstrates how to use the spatiotemporal raster module to compute
coastal management indicators from DTM (Digital Terrain Model) data.

Usage:
    # Run new analysis
    python "spatiotemporal - E02 - coastal-managers-indicators.py"
"""

from pathlib import Path
import environmentaltools.spatiotemporal.raster as raster
from environmentaltools.graphics import spatiotemporal
from loguru import logger
import json

def main():
    logger.info("="*60)
    logger.info("COASTAL MANAGEMENT INDICATORS EXAMPLE")
    logger.info("="*60)

    # Run new analysis
    config_path = Path("src/environmentaltools/data/spatiotemporal/raster/coastal_management/config.json")

    # Cargar configuración desde JSON
    with open(config_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    logger.info(f"Using configuration file: {config_path}")
    logger.info("Starting analysis...")

    # Execute the raster analysis pipeline (saves results automatically)
    raster.analysis(info)

    logger.info("Done!")


if __name__ == '__main__':
    main()

