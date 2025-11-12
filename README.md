# environmentaltools

**environmentaltools** is an open-source Python package for modular environmental management, integrating timeseries analysis, raster-based processing, decision matrices, sensor workflows, and legal-administrative support for solutions to real engineering and environmental problems. Like Python and most of the packages developed by the scientific community, *environmentaltools* is an open-source software. 

It is compound by a list of subpackages that focus on:

**Table 1**. Package name, mean objective and state of packages included in environmentaltools.
| Package name                    | Mean objective                                           | State         | Base installation
|---------------------------------|----------------------------------------------------------| --------------|-----|
| environmentaltools.**common**        | Auxiliary utilities for data handling and processing     | Stable        | **Yes** |
| environmentaltools.**data**         | Download environmental data from various sources (CMEMS) | Download from GitHub| No  |
| environmentaltools.**drone**        | UAV mission planning and flight data generation          | Under develop | No  |
| environmentaltools.**examples**     | Full catalog of examples of each package                 | Download from GitHub        | No  |
| environmentaltools.**graphics**     | Visualization tools for environmental data               | Stable        | **Yes** |
| environmentaltools.**processes**    | Wave modeling and environmental processes                | Stable        | No  |
| environmentaltools.**spatial**      | Geospatial analysis and topography/bathymetry processing | Stable        | No  |
| environmentaltools.**spatiotemporal**| BME and raster-based spatiotemporal analysis            | Under develop | No  |
| environmentaltools.**spectral**     | Spectral analysis (Lomb-Scargle periodogram)             | Under develop | No  |
| environmentaltools.**temporal**     | Time series processing and statistical characterization  | Stable        | **Yes** |

Each module has several dependencies, so it is recommended to partially install the packages required after the creation of a virtual environment. The basic installation comprises the packages given in Table 1 (**basic installation**) To install lonely an extra package the name of the environmentaltools package given in the Table 1 (**package name**) is required. That can be done:

`python -m venv environmentaltools`  
`pip install environmentaltools` # for installing Full Package (common, graphics, temporal, etc.)  
`pip install -e environmentaltools[download]` # for installing download package  
`pip install -e environmentaltools[spatiotemporal]` # for installing spatiotemporal package  
`pip install -e environmentaltools[estuaries,examples]` # for installing both estuaries and examples  

## 📊 LaTeX support for figure rendering

This package uses LaTeX to render text in plots and figures (e.g., axis labels, titles, legends).  
To enable this feature, you must have a LaTeX distribution installed and available in your system `PATH`.

Supported distributions:
- [MiKTeX](https://miktex.org/download) (Windows)
- [TeX Live](https://tug.org/texlive/) (Cross-platform)
- [TinyTeX](https://yihui.org/tinytex/) (Lightweight)

Once installed, make sure the `latex` command is available in your terminal.  
The system will automatically use LaTeX to render text in figures if `text.usetex = True` is enabled.

The tool is developed by Manuel Cobos (https://github.com/mcobosb) as part of the **Environmental Fluid Dynamics (GDFA, https://dinamicambiental.es)** team of the University of Granada. The GDFA wishes a good experience in learning process. Enjoy it!

## Subpackages description

### **data** subpackage
The *data* subpackage provides tools for downloading environmental data from various online sources. It includes automated functions to access and retrieve data from the **Marine Copernicus Service (CMEMS)**, allowing users to download oceanographic variables (sea surface temperature, currents, wave data, etc.) for specific spatial domains, depth ranges, and time periods. This module simplifies the process of obtaining high-quality environmental data for analysis and modeling purposes.

### **drone** subpackage
The *drone* subpackage provides comprehensive tools for **unmanned aerial vehicle (UAV) mission planning** and flight data generation for environmental monitoring applications. It includes functions for scan pattern generation, waypoint optimization, flight time calculation, and KMZ mission file creation for DJI aircraft. The module supports photogrammetry surveys, enables efficient coverage of study areas with configurable overlap parameters, and facilitates mission management through automated batching and preview generation. This tool is particularly useful for aerial data collection in coastal, terrestrial, and aquatic environments.

### **examples** subpackage
In the *example** folder can be found a list of Jupyter Notebooks. Each one described how to run the code and how to use the main functions included in *environmentaltools*.

### **graphics** subpackage
The *graphics* subpackage offers a comprehensive set of visualization tools specifically designed for environmental data. It includes functions to create 2D and 3D plots, spatial maps, time series visualizations, scatter plots for Maximum Dissimilarity Algorithm (MDA) cases, regime diagrams, regression plots, and spatiotemporal representations. The module leverages Matplotlib and specialized colormaps (cmocean) to produce publication-quality figures for scientific communication.

### **processes** subpackage
The *processes* subpackage provides tools for modeling and computing environmental processes, with a focus on **wave modeling** using numerical models like SWAN (Simulating WAves Nearshore) and COPLA. It includes functions to create project databases, manage computational meshes, prepare input files, run simulations, and process model outputs. The module supports wave climate analysis, wave transformation studies, and the assessment of wave-structure interactions.

### **spatial** subpackage
The *spatial* subpackage contains functions for **geospatial analysis** and processing of topographic and bathymetric data. It provides tools to merge land and sea elevation data, perform spatial interpolations, compute distances and nearest neighbors, handle coordinate transformations, and work with raster datasets. The module is designed to facilitate the preparation of spatial data for environmental modeling and analysis applications.

### **spatiotemporal** subpackage
The *spatiotemporal* subpackage implements advanced methods for analyzing data with both spatial and temporal dimensions. It includes the **Bayesian Maximum Entropy (BME)** framework for spatiotemporal estimation and prediction, covariance function modeling (covST), and raster-based analysis tools. The module allows users to estimate environmental variables at unsampled locations and times by integrating hard data, soft data (uncertain information), and prior knowledge through a rigorous probabilistic approach.

### **spectral** subpackage
The *spectral* subpackage provides tools for **spectral analysis** of time series data. It implements the **Lomb-Scargle periodogram** for analyzing unevenly sampled data, allowing users to identify dominant periodicities and frequency components in environmental time series. This is particularly useful for detecting seasonal cycles, tidal signals, and other periodic patterns in irregular datasets where traditional Fourier analysis may not be appropriate.

### **temporal** subpackage
The subpackage *temporal* package aimed at providing users with a friendly, general code to statistically characterize a vector random process (RP) to obtain realizations of it. It is implemented in Python - an interpreted, high-level, object-oriented programming language widely used in the scientific community - and it makes the most of the Python packages ecosystem. Among the existing Python packages, it uses Numpy, which is the fundamental package for scientific computing in Python [["1"]](#1), SciPy, which offers a wide range of optimization and statistics routines [["2"]](#2), Matplotlib [["3"]](#3), that includes routines to obtain high-quality graphics, and Pandas [["4"]](#4) to analyse and manipulate data.

The tools implemented in the package named *temporal* allow to capture the statistical properties of a **non stationary (NS) vector RP** by using **compound or piecewise parametric PMs** to properly describe all the range of values and to **simulate uni- or multivariate time series** with the same random behavior. The statistical parameters of the distributions are assumed to depend on time and are expanded into a Generalized Fourier Series (GFS) [["5"]](#5) in order to reproduce their NS behavior. The applicability of the present approach has been illustrated in several works with different purposes, among others: (i) the observed wave climate variability in the preceding century and expected changes in projections under a climate change scenario [["6"]](#6); (ii) the optimal design and management of an oscillating water column system [["7"]](#7) [["8"]](#8), (iii) the planning of maintenance strategies of coastal structures [["9"]](#9), (iv) the analysis of monthly Wolf sunspot number over a 22 year period [["5"]](#5), and (v) the simulation of estuarine water conditions for the management of the estuary [["10"]](#10).

### **utils** subpackage
The *utils* subpackage contains a collection of **auxiliary utilities** that support the functionality of other modules within the package. It includes functions for data loading and saving in various formats (NetCDF, CSV, pickle), file reading and writing operations, data manipulation and transformation, xarray dataset utilities, and miscellaneous helper functions. The module provides a consistent interface for common operations across the package, improving code reusability and maintainability.

## Author's Note

This project emerged from more than 12 years of work at GDFA across multiple research and engineering projects, where numerous disparate tools were developed—nearly all in Python. The goal was to consolidate these scattered solutions and make both advanced and fundamental environmental analysis methods accessible to a broader audience. By unifying these tools into a cohesive, modular framework, I hope to enable technical teams and researchers to deploy robust environmental workflows without relying on proprietary software. Feedback, suggestions, and contributions are always welcome.

— Manuel