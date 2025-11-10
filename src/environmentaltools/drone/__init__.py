"""Drone utilities for environmental data collection and mission planning.

This module provides tools for drone-based environmental monitoring, including
flight planning, waypoint generation, and mission management for DJI aircraft.

Features:
    - Scan pattern generation for photogrammetry surveys
    - KMZ mission file generation for DJI Pilot 2
    - Waypoint optimization and flight time calculation
    - Mission batching and management
    - Geographic data integration (GeoPackage, CSV)

Submodules:
    build_scan_pattern: Generate scan patterns for aerial surveys
    drone_missions: Create and manage DJI mission files

Examples:
    Basic scan pattern generation:
    
    >>> from environmentaltools.drone import build_scan_pattern
    >>> config = {
    ...     'height': 100,
    ...     'fov_deg': 84,
    ...     'lateral_overlap': 0.8,
    ...     'longitudinal_overlap': 0.8
    ... }
    >>> spacing = build_scan_pattern.calculate_scan_parameters(**config)
    
    KMZ mission creation:
    
    >>> from environmentaltools.drone import drone_missions
    >>> drone_missions.create(
    ...     path="./missions",
    ...     polygon_no="001",
    ...     chunk_size=50,
    ...     template_path="template.kmz"
    ... )

Dependencies:
    Core: pandas, geopandas, shapely
    Visualization: matplotlib, pillow
    Geospatial: pyproj, affine
"""

# Scan pattern generation
from .build_scan_pattern import (
    ground_coverage,
    extract_polygons,
    load_study_area,
    filter_polygons,
    calculate_scan_parameters,
    generate_scan_lines,
    create_waypoints_from_lines,
    save_waypoints_csv,
    save_waypoints_gpkg,
    plot_polygon_flight_plan,
    plot_complete_flight_plan,
    calculate_flight_time,
    process_polygon,
    analysis,
)

# Mission management
from .drone_missions import (
    generate_wpml_from_csv,
    build_kmz_from_template,
    list_dji_dirs,
    create_preview,
    create,
    rename,
)

# Export main functions
__all__ = [
    # Scan pattern generation
    "ground_coverage",
    "extract_polygons", 
    "load_study_area",
    "filter_polygons",
    "calculate_scan_parameters",
    "generate_scan_lines",
    "create_waypoints_from_lines",
    "save_waypoints_csv",
    "save_waypoints_gpkg",
    "plot_polygon_flight_plan",
    "plot_complete_flight_plan",
    "calculate_flight_time",
    "process_polygon",
    "analysis",
    
    # Mission management
    "generate_wpml_from_csv",
    "build_kmz_from_template",
    "list_dji_dirs",
    "create_preview",
    "create",
    "rename",
]