Download module
===============

The **download** module provides utilities for downloading environmental data from various 
online sources including CORDEX climate data, Google Earth Engine, Marine Copernicus (ERA5), 
OpenStreetMap imagery, and Google Maps.

.. automodule:: environmentaltools.download
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


.. currentmodule:: environmentaltools.download

CORDEX Data
-----------

Functions for downloading CORDEX climate model data from ESGF nodes.

.. autosummary::
   :toctree: _autosummary

   query_esgf_catalog
   download_esgf_dataset
   download_with_config

Google Earth Engine
-------------------

Functions for downloading satellite imagery from Google Earth Engine.

.. autosummary::
   :toctree: _autosummary

   initialize_earth_engine
   create_study_area_geometry
   calculate_vegetation_indices
   create_sentinel2_collection
   download_image_with_geemap
   download_single_sentinel2_image
   download_sentinel2_images

Google Maps
-----------

Classes and functions for downloading Google Maps imagery.

Classes
~~~~~~~

.. autosummary::
   :toctree: _autosummary

   GoogleMapsLayers
   GoogleMapDownloader

Functions
~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   download_google_maps_image

OpenStreetMap
-------------

Functions for downloading and visualizing OpenStreetMap imagery.

.. autosummary::
   :toctree: _autosummary

   download_openstreet_map
   create_osm_image
   calculate_extent

Marine Copernicus (ERA5)
------------------------

Classes and functions for downloading ERA5 reanalysis data from the Copernicus Climate Data Store.

Configuration
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   ERA5DataDownloadConfig

Download
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   ERA5DataDownloader

Processing
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   ERA5DataProcessor

Main Functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   download_era5_data
