Spatial Module
==============

.. automodule:: environmentaltools.spatial
   :no-index:

The spatial module provides tools for geospatial analysis and processing of topographic and bathymetric data.

Spatial Analysis
----------------

Data Selection and Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: environmentaltools.spatial

.. autosummary::
   :toctree: _autosummary

   select_data
   interp
   fillna

Profile Analysis
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   normal_profiles
   confidence_interval_2d

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   rotate_coords
   global_to_local_coords
   local_to_global_coords

Geometric Operations
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   continuous_line
   create_polygon
   triangulation

Voronoi Diagrams
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   generate_CVD
   bounded_voronoi
   centroid_region
   in_box

Geospatial Tools
----------------

.. autosummary::
   :toctree: _autosummary

   merge_land_sea
   spatial_mask
.. autofunction:: environmentaltools.spatial.remove_lowland
.. autofunction:: environmentaltools.spatial.merge_sea_sea
.. autofunction:: environmentaltools.spatial.transform_coordinates
