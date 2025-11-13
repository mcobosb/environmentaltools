"""
Coastal Management Indicators Example

This script demonstrates how to use the spatiotemporal raster module to compute
coastal management indicators from DTM (Digital Terrain Model) data.

The script can either:
1. Run a new analysis from config.json
2. Load previously saved results from JSON+NPZ format

Usage:
    # Run new analysis
    python "spatiotemporal - E02 - coastal-managers-indicators.py"
    
    # Load existing results (directory or metadata.json)
    python "spatiotemporal - E02 - coastal-managers-indicators.py" --load "results/indicator_results_20250112_143022"
"""

import os
import sys
from pathlib import Path
import environmentaltools.spatiotemporal.raster as raster
from environmentaltools.graphics import spatiotemporal
from loguru import logger
import json

logger.info("="*60)
logger.info("COASTAL MANAGEMENT INDICATORS EXAMPLE")
logger.info("="*60)

# Check if loading existing results
if len(sys.argv) > 1 and sys.argv[1] == "--load":
    if len(sys.argv) < 3:
        logger.error("Please provide path to results file: --load <path>")
        sys.exit(1)
    
    results_file = Path(sys.argv[2])
    logger.info(f"Loading results from: {results_file}")
    results = raster.load_results(results_file)
    
else:
    # Run new analysis
    cwd = os.path.dirname(os.path.abspath(__file__))
    config_path = Path("C:/Users/m_cob/Desktop/test_raster_tool/coastal_management/config.json")

    # Cargar configuración desde JSON
    with open(config_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    logger.info(f"Using configuration file: {config_path}")
    logger.info("Starting analysis...")

    # Execute the raster analysis pipeline (saves results automatically)
    info, results = raster.analysis(info)

# Visualize results
logger.info("="*60)
logger.info("VISUALIZING RESULTS")
logger.info("="*60)

for index in results.keys():
    logger.info(f"Visualizing {len(results[index])} result(s) for index '{index}'")
    for i, result in enumerate(results[index]):
        logger.info(f"  - Simulation {i+1}: {len(result[0])} contours found")
        spatiotemporal.plot_presence_boundary(result[0], result[1])

logger.info("Done!")

