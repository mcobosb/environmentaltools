Installation
============


**environmentaltools** is an open-source Python package for modular environmental management, integrating timeseries analysis, raster-based processing, decision matrices, sensor workflows, and legal-administrative support for solutions to real engineering and environmental problems. Like Python and most of the packages developed by the scientific community, *environmentaltools* is an open-source software. 

It is compound by a list of subpackages that focus on:

.. table:: **Table 1**. Package name, mean objective and state of packages included in environmentaltools.
   :align: left

   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | Package name                        | Mean objective                                           | State         | Base installation |
   +=====================================+==========================================================+===============+===================+
   | environmentaltools.**common**       | Auxiliary utilities for data handling and processing     | Stable        | **Yes**           |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**data**         | Download environmental data from various sources (CMEMS) | Stable        | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**estuaries**    | Saint-Venant equations for estuarine dynamics            | Under develop | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**examples**     | Full catalog of examples of each package                 | Stable        | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**graphics**     | Visualization tools for environmental data               | Stable        | **Yes**           |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**processes**    | Wave modeling and environmental processes                | Stable        | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**spatial**      | Geospatial analysis and topography/bathymetry processing | Stable        | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**spatiotemporal** | BME and raster-based spatiotemporal analysis           | Under develop | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**spectral**     | Spectral analysis (Lomb-Scargle periodogram)             | Under develop | No                |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+
   | environmentaltools.**temporal**     | Time series processing and statistical characterization  | Stable        | **Yes**           |
   +-------------------------------------+----------------------------------------------------------+---------------+-------------------+

Each module has several dependencies, so it is recommended to partially install the packages required after the creation of a virtual environment. The basic installation comprises the packages given in Table 1 (**basic installation**) To install lonely an extra package the name of the environmentaltools package given in the Table 1 (**package name**) is required. That can be done:

.. code-block:: bash

   python -m venv environmentaltools  
   pip install . # for installing Basic packages (common, graphics, temporal)  
   pip install -e .[download] # for installing download package  
   pip install -e .[spatiotemporal] # for installing spatiotemporal package  
   pip install -e .[estuaries, examples] # for installing both estuaries and examples  
   pip install -e .[all] # for installing the full repository  

LaTeX support for figure rendering
-----------------------------------

This package uses LaTeX to render text in plots and figures (e.g., axis labels, titles, legends).  
To enable this feature, you must have a LaTeX distribution installed and available in your system ``PATH``.

**Supported distributions:**

- `MiKTeX <https://miktex.org/download>`_ (Windows)
- `TeX Live <https://tug.org/texlive/>`_ (Cross-platform)
- `TinyTeX <https://yihui.org/tinytex/>`_ (Lightweight)

Once installed, make sure the ``latex`` command is available in your terminal.  
The system will automatically use LaTeX to render text in figures if ``text.usetex = True`` is enabled.