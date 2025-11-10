Drone Module
============

.. automodule:: environmentaltools.drone
   :no-index:

The drone module provides comprehensive tools for unmanned aerial vehicle (UAV) mission planning, 
scan pattern generation, and flight data management for environmental monitoring applications.

Scan Pattern Generation
-----------------------

Flight Planning and Coverage Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: environmentaltools.drone

.. autosummary::
   :toctree: _autosummary

   ground_coverage
   calculate_scan_parameters
   calculate_flight_time

Polygon and Study Area Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   extract_polygons
   load_study_area
   filter_polygons

Waypoint Generation
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   generate_scan_lines
   create_waypoints_from_lines
   process_polygon
   analysis

Data Export and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   save_waypoints_csv
   save_waypoints_gpkg
   plot_polygon_flight_plan
   plot_complete_flight_plan

Mission Management
------------------

DJI Mission File Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   generate_wpml_from_csv
   build_kmz_from_template
   create
   rename

Mission Organization
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   list_dji_dirs
   create_preview

Integration with Other Modules
-------------------------------

The drone module integrates seamlessly with other environmentaltools modules:

**Spatial Module Integration**
   Use spatial analysis tools for pre-flight area assessment and post-flight data processing.

**Common Module Integration**
   Leverage common utilities for coordinate transformations and data I/O operations.

**Graphics Module Integration**
   Create publication-ready flight plan visualizations and mission summary plots.

Dependencies
------------

**Required**
   - pandas: DataFrame operations and CSV handling
   - geopandas: Geospatial data processing
   - shapely: Geometric operations
   - matplotlib: Plotting and visualization
   - pillow: Image processing for mission previews

**Optional**
   - pyproj: Coordinate reference system transformations
   - affine: Geometric transformations