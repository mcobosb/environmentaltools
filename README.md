# environmentaltools

**environmentaltools** is an open-source Python package for modular environmental management, integrating timeseries analysis, raster-based processing, decision matrices, sensor workflows, and legal-administrative support for solutions to real engineering and environmental problems. Like Python and most of the packages developed by the scientific community, *environmentaltools* is an open-source software. 

It is compound by a list of subpackages that focus on:

| Módulo                          | Función principal                                        | Estado        |
|---------------------------------|----------------------------------------------------------|---------------|
| environmentaltools.data         | Download environmental data from various sources (CMEMS) | Estable       |
| environmentaltools.estuaries    | Saint-Venant equations for estuarine dynamics            | En desarrollo |
| environmentaltools.graphics     | Visualization tools for environmental data               | Estable       |
| environmentaltools.processes    | Wave modeling and environmental processes (SWAN, COPLA)  | Estable       |
| environmentaltools.spatial      | Geospatial analysis and topography/bathymetry processing | Estable       |
| environmentaltools.spatiotemporal| BME and raster-based spatiotemporal analysis            | Estable       |
| environmentaltools.spectral     | Spectral analysis (Lomb-Scargle periodogram)             | Estable       |
| environmentaltools.temporal     | Time series processing and statistical characterization  | En desarrollo |
| environmentaltools.utils        | Auxiliary utilities for data handling and processing     | Estable       |


## Subpackages description

### **data** subpackage
The *data* subpackage provides tools for downloading environmental data from various online sources. It includes automated functions to access and retrieve data from the **Marine Copernicus Service (CMEMS)**, allowing users to download oceanographic variables (sea surface temperature, currents, wave data, etc.) for specific spatial domains, depth ranges, and time periods. This module simplifies the process of obtaining high-quality environmental data for analysis and modeling purposes.

### **estuaries** subpackage
The *estuaries* subpackage implements numerical solutions for the **Saint-Venant equations** applied to estuarine dynamics. It provides tools to simulate hydrodynamic processes in estuaries, including water level variations, flow velocities, salinity transport, and density computations. The module is designed to study the consequences of natural and anthropogenic actions in estuarine environments, supporting management and decision-making processes for coastal water bodies.

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

In the **example folder** can be found a list of Jupyter Notebooks. Each one described how to run the code and how to use the main functions included in *environmentaltools*.

The **Environmental Fluid Dynamics** team of the University of Granada whishes a good experience in learning process. Enjoy it!

### **utils** subpackage
The *utils* subpackage contains a collection of **auxiliary utilities** that support the functionality of other modules within the package. It includes functions for data loading and saving in various formats (NetCDF, CSV, pickle), file reading and writing operations, data manipulation and transformation, xarray dataset utilities, and miscellaneous helper functions. The module provides a consistent interface for common operations across the package, improving code reusability and maintainability.


## References
<a id="1">[1]</a> 
Harris, Charles R. and Millman, K. Jarrod and
    van der Walt, Stéfan J and Gommers, Ralf and
    Virtanen, Pauli and Cournapeau, David and
    Wieser, Eric and Taylor, Julian and Berg, Sebastian and
    Smith, Nathaniel J. and Kern, Robert and Picus, Matti and
    Hoyer, Stephan and van Kerkwijk, Marten H. and
    Brett, Matthew and Haldane, Allan and
    Fernández del Río, Jaime and Wiebe, Mark and
    Peterson, Pearu and Gérard-Marchant, Pierre and
    Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and
    Abbasi, Hameer and Gohlke, Christoph and
    Oliphant, Travis E. (2020). 
Array programming with {NumPy}.
Nature.

<a id="2">[2]</a> 
Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
  Haberland, Matt and Reddy, Tyler and Cournapeau, David and
  Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
  Bright, Jonathan and {van der Walt}, Stéfan J. and
  Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
  Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
  Kern, Robert and Larson, Eric and Carey, C J and
  Polat, Ilhan and Feng, Yu and Moore, Eric W. and
  {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
  Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
  Harris, Charles R. and Archibald, Anne M. and
  Ribeiro, Antonio H. and Pedregosa, Fabian and
  {van Mulbregt}, Paul and {SciPy 1.0 Contributors} (2020).
{{SciPy} 1.0: Fundamental Algorithms for Scientific
Computing in Python}.
Nature Methods.
  
<a id="3">[3]</a> 
John D. Hunter.
Matplotlib: A 2D Graphics Environment.
Computing in Science & Engineering.

<a id="4">[4]</a> 
McKinney, Wes and others (2010).
Data structures for statistical computing in python.
Proceedings of the 9th Python in Science Conference.

<a id="5">[5]</a> 
Cobos, M. and Otíñar, P. and Magaña, P. and Baquerizo, A. (2021)
A method to characterize and simulate climate, Earth or environmental vector random processes.
Submitted to Probabilistic Engineering and Mechanics.

<a id="6">[6]</a> 
Lira-Loarca, Andrea Lira and Cobos, Manuel and Besio, Giovanni and Baquerizo, Asunción (2021).
Projected wave climate temporal variability due to climate change.
Stochastic Environmental Research and Risk Assessment.

<a id="7">[7]</a> 
Jalón, María L and Baquerizo, Asunción and Losada, Miguel A (2016).
Optimization at different time scales for the design and management of an oscillating water column system.
Energy.

<a id="8">[8]</a> 
López-Ruiz, Alejandro and Bergillos, Rafael J and Lira-Loarca, Andrea and Ortega-Sánchez, Miguel (2018).
A methodology for the long-term simulation and uncertainty analysis of the operational lifetime performance of wave energy converter arrays.
Energy.

<a id="9">[9]</a> 
Lira-Loarca, Andrea and Cobos, Manuel and Losada, Miguel Ángel and Baquerizo, Asunción (2020).
Storm characterization and simulation for damage evolution models of maritime structures.
Coastal Engineering.

<a id="10">[10]</a> 
Cobos, Manuel (2020).
A model to study the consequences of human actions in the Guadalquivir River Estuary.
Tesis Univ. Granada.
