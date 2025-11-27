"""
Spatiotemporal Raster Analysis Module
====================================

This module provides comprehensive tools for spatiotemporal raster analysis in 
environmental applications, with a focus on coastal management and marine spatial 
analysis. The module handles NetCDF-based environmental data cubes and generates 
binary masks for threshold-based analysis.

Main Components
---------------

**Configuration Management**
    - JSON-based configuration file loading and validation
    - Input data verification and consistency checking
    - Output directory structure creation

**Temporal Analysis**
    - Temporal difference calculation between elevation layers
    - Change statistics computation across space and time
    - Multi-simulation temporal pattern analysis

**Binary Matrix Generation**
    - Threshold-based binary mask creation
    - Spatiotemporal data cube processing
    - NetCDF output generation with metadata

**Preprocessing and Refinement**
    - Data preprocessing for spatial analysis
    - Grid refinement and coordinate transformation
    - Level data processing and statistics

Functions
---------
calculate_temporal_differences : Analyze temporal changes in elevation data
check_inputs : Validate input data and setup processing parameters
post_treatment : Perform preprocessing steps for analysis
binary_matrix : Generate binary masks from threshold comparison
analysis : Execute complete spatiotemporal raster analysis workflow

Examples
--------
Basic usage for coastal management analysis:

    >>> from pathlib import Path
    >>> from environmentaltools.spatiotemporal.raster import analysis
    >>> 
    >>> # Setup configuration file path
    >>> config_path = Path("config/coastal_analysis.json")
    >>> 
    >>> # Execute complete analysis workflow
    >>> results = analysis(config_path)
    >>> print(f"Processed {len(results['datacube_filenames'])} simulations")

The configuration file should contain:
    - project: Input/output paths and simulation parameters
    - temporal: Time-related processing settings
    - parameters: Analysis parameters including refinement options
    - seasons: Seasonal definitions for temporal analysis

Notes
-----
This module is designed for processing large spatiotemporal datasets and 
includes memory usage monitoring and optimization features. It supports
multiple simulation scenarios and analysis indices for comprehensive
environmental assessment.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import json
from loguru import logger

from datetime import datetime
from time import time

import geopandas as gpd

from shapely import LineString
from pyproj import CRS

from skimage.measure import find_contours
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
import psutil
from environmentaltools.spatiotemporal import indicators


GPKG_ACTIVE = False # Set to True if Fiona and GeoPackage support are available (for GeoPackage output instead of Shapefile)


def largest_region_from_field(batches):
    """
    Selects the largest region from a continuous field and returns its contour.

    Parameters
    ----------
    batches : tuple
        Tuple containing:
            - field: 2D ndarray of continuous values.
            - refinement: bool, whether to use refined contour extraction.

    Returns
    -------
    contours : list of ndarray
        List of (y, x) coordinates of the interpolated contour for the largest region.
    """
    field, refinement = batches
    if not refinement:
        # 1. Create binary mask: True where field <= 0, False otherwise
        field = field <= 0
        contours = find_contours(field.astype(bool), level=0)
    else:
        # Use float field for refined contour extraction
        contours = find_contours(field.astype(float), level=0)

    # Filter contours to keep only the one corresponding to the largest region
    lengths = [c.shape[0] for c in contours]
    idx_long = int(np.argmax(lengths))
    contours = [contours[idx_long]]

    return contours


def preprocess(ds):
    """
    Expand the dataset by adding a 'simulation' dimension using the 'simulation' attribute.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset, expected to have a 'simulation' attribute.

    Returns
    -------
    xarray.Dataset
        Dataset with an added 'simulation' dimension labeled by the simulation ID.
    """
    sim_id = ds.attrs.get("simulation", "unknown")
    return ds.expand_dims({"simulation": [sim_id]})


def check_inputs(info):
    """
    Validate and prepare input configuration for marine spatial analysis processing.

    Performs comprehensive validation of input parameters, file paths, and configuration
    settings required for the marine tools spatial analysis workflow. Sets up default
    values, creates necessary directory structures, and validates data availability.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing project parameters, file paths, and
        processing settings. Expected structure and keys include:

        - "project" : dict
            Project-level configuration with keys:
                - "name" : str
                    Name of the project/region
                - "no_sims" : int
                    Number of simulations to process
                - "initial_year" : int
                    Initial year of the simulation
                - "final_year" : int
                    Final year of the simulation
                - "input_path" : str or Path
                    Path to the input data directory (e.g., DTM/NetCDF files)
                - "output_path" : str or Path
                    Path to the output results directory
                - "crs" : str
                    Coordinate Reference System (e.g., "EPSG:25830")
                - "refinement" : int, optional
                    Refinement flag (1 for True, 0 for False)
                - "n_workers" : int, optional
                    Number of parallel workers
                - "index" : str
                    Name of the indicator/index to process (e.g., "directional_influence")

        - "temporal" : dict
            Temporal configuration, including:
                - "seasons" : list of str
                    List of season keys (e.g., ["TA", "TB", "AN"])
                - "TA", "TB", ... : list of int
                    Lists of months for each season

        - Indicator-specific keys (e.g., "presence_boundary", "influence_extent", "directional_influence") : dict
            Each indicator contains:
                - "variable" : str
                    Name of the variable in NetCDF files (e.g., "elevation")
                - "temporal" : str or list
                    Temporal variable or thresholds
                - "percentiles" : list of str or int, optional
                    Percentiles to compute (if applicable)
                - "horizon_times" : list of str, optional
                    List of horizon times (for some indices)
                - "return_periods" : list of int, optional
                    Return periods (for some indices)

    Returns
    -------
    None
        Function modifies the input dictionary in-place.

    Raises
    ------
    FileNotFoundError
        If required input files or project do not exist.
    ValueError
        If fld_portion parameter is not a positive integer.

    Notes
    -----
    The function performs the following validations and setup:
    1. Verifies existence of input DTM/NetCDF files
    2. Checks simulation project and catalogs available files
    3. Sets default values for optional parameters
    4. Defines seasonal month groupings if not provided
    5. Creates auxiliary output directory structure
    6. Validates flood portion parameter
    """
    # Check if project configuration exists
    if not "project" in info.keys():
        raise ValueError("Project configuration not specified.")

    if not "input_path" in info["project"].keys():
        raise ValueError("Input path not specified.")
    
    info["project"]["input_path"] = Path(info["project"]["input_path"]).expanduser()
    
    if not "no_sims" in info["project"].keys():
        logger.info("Number of simulations not specified, defaulting to 1.")
        info["project"]["no_sims"] = 1
    
    if info["project"]["no_sims"] < 1:
        raise ValueError("Number of simulations must be at least 1.")
    
    if not info["project"]["input_path"].exists():
        raise FileNotFoundError(f"Input datacube path does not exist: {info['project']['input_path']}")

    # Obtain all NetCDF files in the input path
    info["datacube_filenames"] = [str(file) for file in info["project"]["input_path"].rglob("*.nc")]
    
    # Limit to the number of simulations specified in the configuration
    if len(info["datacube_filenames"]) < info["project"]["no_sims"]:
        raise ValueError(f"Number of NetCDF files ({len(info['datacube_filenames'])}) is lower than the number of simulations ({info['project']['no_sims']}).")
    
    info["datacube_filenames"] = info["datacube_filenames"][:info["project"]["no_sims"]]

    # Get coordinate metadata from NetCDF file and estimate memory usage for processing
    with xr.open_dataset(info["datacube_filenames"][0]) as ds:
        logger.info("="*60)
        # Determine the shape from the first data variable (DataArray)
        var_name = list(ds.data_vars.keys())[0]
        data_var = ds[var_name]
        info["dimensions"] = [info["project"]["no_sims"], *data_var.shape]
        logger.info(f"Data cube dimensions and coordinates: {info['dimensions']}")

        # Estimate memory usage for the data stack
        mem = psutil.virtual_memory()
        total_mem = mem.total / (1024**2)
        logger.info(f"System total memory: {total_mem} MB")

        data_in_memory = np.ndarray([info["project"]["no_sims"], *data_var.shape], dtype=np.float32)
        array_mem = data_in_memory.nbytes / (1024**2)
        logger.info(f"Allocated binary mask and data matrix size: {array_mem:.2f} MB/{array_mem/total_mem*100:.2f}% of system memory.")
        if array_mem > total_mem * 0.8:
            logger.warning("Allocated binary mask and data matrix exceeds 80% of system memory. Consider reducing data size or using chunked processing.")
        del data_in_memory
        logger.info("="*60)

    if "index" not in info["project"].keys():
        logger.error("Index not specified in configuration.")
        raise ValueError("Index not specified in configuration.")
    
    # Set project index in the info dictionary
    info["index"] = info["project"]["index"]
    
    # Check that the specified index is valid
    if info["index"] not in ["presence_boundary", "influence_extent", "threshold_exceedance_frequency", "permanently_affected_zone",
        "mean_representative_value", "return_period_based_extreme_value", "spatial_change_rate", "functional_area_loss",
        "critical_boundary_retreat", "neighborhood", "neighborhood_gradient_influence", "environmental_convergence",
        "neighborhood_polarization", "local_persistence", "environmental_risk", "directional_influence",
        "multivariate_neighborhood_synergy", "spatiotemporal_coupling", "multivariate_threshold_exceedance",
        "directional_co_evolution", "multivariate_persistence", "multivariate_recovery"]:
        raise ValueError(f"Invalid index specified: {info["index"]}")

    # TODO: Define the 'contour' flag for the rest of the indices if needed
    if (info["index"] == "presence_boundary") | (info["index"] == "influence_extent"):
        info["contour"] = True
    else:
        info["contour"] = False
    
    # Additional checks for specific indices (e.g., presence_boundary, directional_influence)
    if (info["index"] == "presence_boundary") | (info["index"] == "directional_influence"):
        # Obtain all CSV level files in the input path
        info["level_filenames"] = [str(file) for file in info["project"]["input_path"].rglob("*.csv")]

        # Limit to the number of simulations specified in the configuration
        if len(info["level_filenames"]) < info["project"]["no_sims"]:
            raise ValueError(f"Number of level files ({len(info['level_filenames'])}) is lower than the number of simulations ({info['project']['no_sims']}).")

        info["level_filenames"] = info["level_filenames"][:info["project"]["no_sims"]]

        # Verify date consistency between NetCDF and CSV files
        # This ensures that all dates in the NetCDF time dimension have corresponding data in the CSV
        # It also ensures that required variables are present in the level files
        for j, filename in enumerate(info["level_filenames"]):
            levels = pd.read_csv(
                            filename,
                            sep=",",
                            index_col=0,
                        )
            
            # Get time metadata from NetCDF file for date comparison
            with xr.open_dataset(info["datacube_filenames"][j]) as ds:
                netcdf_dates = pd.to_datetime(ds.time.values)
                    
            # Convert CSV index to datetime for comparison with NetCDF dates
            csv_dates = pd.to_datetime(levels.index)
            
            # Check that all NetCDF dates are present in the CSV file
            missing_dates = []
            for date in netcdf_dates:
                if date not in csv_dates:
                    missing_dates.append(date)
            
            if missing_dates:
                raise ValueError(f"NetCDF file {info['datacube_filenames'][j]} contains dates not found in CSV file {filename}: {missing_dates}")
            
            # Check for extra dates in CSV (informational only, not required for processing)
            extra_dates = []
            for date in csv_dates:
                if date not in netcdf_dates:
                    extra_dates.append(date)
            
            if extra_dates:
                logger.warning(f"CSV file {filename} contains {len(extra_dates)} extra dates not in NetCDF: {extra_dates[:5]}{'...' if len(extra_dates) > 5 else ''}")

            # Check that the required variable from the configuration is present in the NetCDF file
            if info[info["index"]]["variable"] not in ds.data_vars:
                raise ValueError(f"Variable {info[info["index"]]["variable"]} from project configuration not found in NetCDF file {info['datacube_filenames'][j]}.")
                
            # Check that the required temporal variable from the configuration is present in the CSV file
            if info[info["index"]]["temporal"] not in levels.columns:
                raise ValueError(f"Variable {info[info["index"]]["temporal"]} from project configuration not found in CSV file {filename}.")

        logger.info(f"Data verification successful for simulations.")
        logger.info(f"Dates [{len(netcdf_dates)}], {info[info["index"]]["variable"]} and {info[info["index"]]["temporal"]} found in NetCDF and CSV files, respectively.")

        info["dates"] = csv_dates

        # Check if percentiles are specified, otherwise set default
        if not "percentiles" in info[info["index"]].keys():
            info["percentiles"] = ["50"]
            logger.info("Percentiles not specified, defaulting to ['50'].")
        else:
            info["percentiles"] = [str(p) for p in info[info["index"]]["percentiles"]]
    
    if info["index"] == "influence_extent":
        try:
            ds = xr.open_mfdataset(
                info["datacube_filenames"],
                combine="nested",
                concat_dim="simulation",
                preprocess=preprocess
            )
            info["dates"] = pd.to_datetime(ds.time.values)
            if info[info["index"]]["variable"] not in list(ds.data_vars.keys()):
                raise ValueError(f"Variable {info[info["index"]]['variable']} for influence extent index not found in NetCDF files.")
        except: 
            raise ValueError("Could not read dates from NetCDF files for influence extent index.")
        
        # Check that all specified horizon times are present in the DTM dates
        if "horizon_times" in info["influence_extent"].keys():
            for horizon_time in info["influence_extent"]["horizon_times"]:
                if pd.to_datetime(horizon_time) not in info["dates"]:
                    raise ValueError(
                        "Horizon time %s do not match available dates.", horizon_time
                    )
            info["influence_extent"]["horizon_times"] = [pd.to_datetime(ht) for ht in info["influence_extent"]["horizon_times"]]

    # Check that all return periods are positive integers
    if info["index"] == "return_period_based_extreme_value":
        if not "return_periods" in info["return_period_based_extreme_value"].keys():
            raise ValueError("Return periods not specified for return period based extreme value index.")
        else:
            for rp in info["return_period_based_extreme_value"]["return_periods"]:
                if rp <= 0:
                    raise ValueError(f"Return period {rp} is not positive.")

    # Create the output directory structure for results
    info["project"]["output_files"] = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(info["project"]["output_path"]) / f"indicator_results_{timestamp}"
    info["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check mesh size and alert about memory usage
    # Get coordinate metadata from NetCDF file
    try:
        ds = xr.open_mfdataset(
            info["datacube_filenames"],
            combine="nested",
            concat_dim="simulation",
            preprocess=preprocess
        )

        x_coords, y_coords = ds.x.values, ds.y.values
        # Store coordinate arrays for later conversion of contour pixel indices to real-world coordinates
        info.setdefault("coords", {})
        info["coords"]["x"] = x_coords.tolist()
        info["coords"]["y"] = y_coords.tolist()

        # Determine CRS: prefer explicit config value, otherwise try to detect from dataset
        info["coords"]["crs"] = info.get("project", {}).get("crs")
        logger.info(f"Using CRS: {info['coords']['crs']}")
    except:
        raise ValueError("Could not read coordinate metadata from NetCDF files.")

    # Check and set the months for high and low seasons if not provided
    if not "seasons" in info["temporal"]:
        info["seasons"] = {
            "AN": "annual",
            "TA": [4, 5, 6, 7, 8, 9],
            "TB": [1, 2, 3, 10, 11, 12],
        }
    
    # Check and set the refinement parameter (True/False)
    if "refinement" in info["project"].keys():
        if info["project"]["refinement"] == 1:
            info["refinement"] = True
        else:
            info["refinement"] = False
    else:
        info["refinement"] = False
    
    # Determine parallelism parameters (number of workers) from info or use defaults
    if "n_workers" not in info.keys():
        default_workers = max(1, int(multiprocessing.cpu_count()/2) - 1)
        info["n_workers"] = default_workers
    else:
        default_workers = multiprocessing.cpu_count()
        if info.get("n_workers") > default_workers:
            raise ValueError(f"Requested n_workers {info.get('n_workers')} exceeds maximum {default_workers}.")

    return


def compute_contours(continuous_stack, info):
    """
    Compute the largest connected contour for each time slice in a 3D raster stack.

    This function processes a stack of 2D continuous raster fields (e.g., thresholded or probability maps)
    and extracts the largest connected region (contour) from each time slice. The extraction is performed
    in parallel for efficiency. The refinement parameter is passed to the region extraction function.

    Parameters
    ----------
    continuous_stack : np.ndarray
        3D array of shape (T, Y, X), where T is the number of time slices, and Y, X are spatial dimensions.
        Each slice continuous_stack[t] is a 2D field to be processed.
    info : dict
        Configuration dictionary containing analysis parameters. Must include:
            - "refinement" : bool
                Whether to apply grid refinement in contour extraction.
            - "n_workers" : int
                Number of parallel workers to use for processing.

    Returns
    -------
    contours : list
        List of extracted contours (one per time slice), as returned by largest_region_from_field.
    """
    start_time = time()
    # Number of time slices to process (T axis of the stack)
    contours = []
    n_to_process = int(continuous_stack.shape[0])

    # Build a list of per-slice arguments for largest_region_from_field.
    # Each element is a tuple (field_2d, refinement_flag)
    slice_args = []
    for t in range(n_to_process):
        field2d = continuous_stack[t]
        # Each field2d is a time slice to be processed for contour extraction
        slice_args.append((field2d, info["refinement"]))

    # Process slices in parallel and report progress with ETA (estimated time of arrival)
    total_slices = len(slice_args)
    with ProcessPoolExecutor(max_workers=info["n_workers"]) as ex:
        for slice_idx, slice_res in enumerate(ex.map(largest_region_from_field, slice_args), start=1):
            contours.append(slice_res)
            elapsed = time() - start_time
            print(
                f"Slice {slice_idx}/{total_slices}"
                f", elapsed {elapsed:.1f}s.",
                end="\r", flush=True,
            )
    return contours


def analysis(info=None):
    """
    Execute the complete spatiotemporal raster analysis workflow.

    This is the main entry point for the spatiotemporal raster processing pipeline. It orchestrates
    configuration loading, input validation, simulation and index processing, and output generation.
    The function supports multi-simulation and multi-index workflows for environmental and coastal analysis.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing all project parameters, input/output paths, and analysis settings.
        Expected structure matches the validated configuration (see check_inputs docstring).

    Returns
    -------
    info : dict
        Updated configuration dictionary, including processed results, output file paths, and metadata.

    Notes
    -----
    The analysis workflow includes:
        1. Configuration loading and validation
        2. Input data verification and preprocessing
        3. Binary matrix and contour generation for each simulation and index
        4. NetCDF output file creation with metadata
        5. Progress logging and error handling

    The function processes all simulations and analysis indices as specified in the configuration,
    creating separate output files for each combination. Results and metadata are stored in the returned info dict.

    Examples
    --------
    >>> from pathlib import Path
    >>> import json
    >>> with open("config/coastal_analysis.json") as f:
    ...     config = json.load(f)
    >>> results = analysis(config)
    >>> print(f"Processed {len(results['datacube_filenames'])} simulations")
    """
    logger.info("="*60)
    logger.info("STARTING SPATIOTEMPORAL RASTER ANALYSIS")
    logger.info("="*60)
    
    # Validate input data and setup processing parameters
    check_inputs(info)
     
    # Process each analysis index specified in configuration
    # (for multi-index workflows, loop over indices here)
    logger.info(f"Processing project index: {info['project']['index']}.")
    
    if info["index"] == "presence_boundary":
        # Allocate a continuous data stack (original variable values) to use for percentile calculations
        data_stack = np.ndarray(info["dimensions"], dtype=np.float32)
        # Process each simulation for the current index (loop over simulations)
        for sim_no in range(info["project"]["no_sims"]):
            # Log simulation progress (show fraction completed for user feedback)
            print(f"Processing simulation {sim_no + 1}/{info['project']['no_sims']} for index {info['project']['index']}.", end="\r", flush=True)
            
            # Load spatiotemporal data cube for current simulation (NetCDF file)
            file_path = info["datacube_filenames"][sim_no]

            # Extract the specified environmental variable from NetCDF file
            data_cube = xr.open_dataset(file_path)[info[info["index"]]["variable"]]
            # Store continuous values for later interpolation (time, y, x)
            data_stack[sim_no, :, :, :] = data_cube.values

            # Load threshold levels from CSV file for current simulation
            levels = pd.read_csv(
                info['level_filenames'][sim_no],
                sep=",",
                index_col=0,
            )
            for j, date in enumerate(levels.index):
                data_stack[sim_no, j, :, :] -= levels.loc[date, info[info["index"]]["temporal"]]
        
        logger.info("="*60)
        logger.info("OBTAINING CONTOURS")
        logger.info("="*60)

        contours = [[] for _ in info['percentiles']]
        for i_perc, p in enumerate(info['percentiles']):
            # Compute the continuous percentile stack from elevation data across the simulation axis (axis=0)
            continuous_stack = np.percentile(data_stack, 100-float(p), axis=0)

            # Compute contours for the current percentile
            contours[i_perc] = compute_contours(continuous_stack, info)
            logger.info(f"Completed percentile {p}%.")
                

    if info["project"]["index"] == "influence_extent":
        contours = [[] for _ in info["influence_extent"]["horizon_times"]]
        
        for i_ht, horizon_time in enumerate(info["influence_extent"]["horizon_times"]):
            # Allocate a continuous data stack (original variable values) to use for percentiles
            len_times = np.sum(info["dates"] <= horizon_time)
            data_stack = np.ndarray((info["dimensions"][0], len_times, info["dimensions"][2], info["dimensions"][3]), dtype=np.float32)

            # Process each simulation for the current index
            for sim_no in range(info["project"]["no_sims"]):
                # Log simulation progress (show fraction completed)
                print(f"Processing simulation {sim_no + 1}/{info['project']['no_sims']} for index {info['project']['index']}.", end="\r", flush=True)
                
                # Load spatiotemporal data cube for current simulation
                file_path = info["datacube_filenames"][sim_no]

                # Extract the specified environmental variable from NetCDF file
                data_cube = xr.open_dataset(file_path)[info[info["index"]]["variable"]]

                dates = [d for d in info["dates"] if horizon_time >= d]
                data_cube_sel = data_cube.sel(time=dates)
                data_stack[sim_no, :, :, :] = data_cube_sel.values - info["influence_extent"]["temporal"][i_ht]
                
            if i_ht == 0:
                logger.info("="*60)
                logger.info(f"OBTAINING CONTOURS")
                logger.info("="*60)

            data_stack = np.max(data_stack, axis=0)

            # Compute contours for the current percentile
            contours[i_ht] = compute_contours(data_stack, info)                
            logger.info(f"Completed horizon time {horizon_time}.")

    if info["project"]["index"] == "directional_influence":
        # Allocate a continuous data stack (original variable values) to use for percentiles
        data_stack = np.ndarray(info["dimensions"], dtype=np.float32)
        # Process each simulation for the current index
        for sim_no in range(info["project"]["no_sims"]):
            # Log simulation progress (show fraction completed)
            print(f"Processing simulation {sim_no + 1}/{info['project']['no_sims']} for index {info['project']['index']}.", end="\r", flush=True)
            
            # Load spatiotemporal data cube for current simulation
            file_path = info["datacube_filenames"][sim_no]

            # Extract the specified environmental variable from NetCDF file
            data_cube = xr.open_dataset(file_path)[info[info["index"]]["variable"]]
            # Store continuous values for later interpolation (time, y, x)
            data_stack[sim_no, :, :, :] = data_cube.values

            # Load threshold levels from CSV file for current simulation
            levels = pd.read_csv(
                info['level_filenames'][sim_no],
                sep=",",
                index_col=0,
            )
            for j, date in enumerate(levels.index):
                data_stack[sim_no, j, :, :] -= levels.loc[date, info[info["index"]]["temporal"]]
        
        results = [[] for _ in info["percentiles"]]
        for i_perc, p in enumerate(info['percentiles']):
            # Compute the continuous percentile stack from elevation data across the simulation axis (axis=0)
            continuous_stack = np.percentile(data_stack, 100-float(p), axis=0)
            angle_map, magnitude_map = indicators.directional_influence(continuous_stack)
            results[i_perc] = [angle_map, magnitude_map]

    if info["contour"]:
        save_contours(contours, info)
    else:
        save_arrays_to_netcdf(results, info)
    
    return

def obtain_geometry(contour, info):
    """
    Convert a contour represented by pixel indices to a real-world geometry.

    Maps a sequence of (row, col) pixel indices from a raster contour to real-world coordinates
    using the x and y coordinate arrays provided in the info dictionary. Returns a LineString
    geometry representing the contour in spatial coordinates.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 2) containing (row, col) pixel indices of the contour.
    info : dict
        Configuration dictionary containing coordinate arrays under info['coords']['x'] and info['coords']['y'].

    Returns
    -------
    geom : shapely.geometry.LineString
        LineString geometry representing the contour in real-world coordinates.
    """
    # contour is an (N,2) array with (row, col)
    # Attempt to map pixel indices to real-world coordinates
    x_coords = np.asarray(info.get("coords", {}).get("x", []))
    y_coords = np.asarray(info.get("coords", {}).get("y", []))

    nx = x_coords.size
    ny = y_coords.size
    col_idx = np.arange(nx)
    row_idx = np.arange(ny)
    row_vals = contour[:, 0]
    col_vals = contour[:, 1]
    xs = np.interp(col_vals, col_idx, x_coords)
    ys = np.interp(row_vals, row_idx, y_coords)
    coords_list = list(zip(xs.astype(float).tolist(), ys.astype(float).tolist()))

    geom = LineString(coords_list)
    return geom


def save_layer(records, info):
    """
    Save a set of geometry records as a spatial layer (GeoPackage or Shapefile).

    Converts a list of geometry records into a GeoDataFrame, assigns the correct CRS,
    and writes the layer to disk. The output format is determined by the GPKG_ACTIVE flag:
    if True, writes to a GeoPackage (GPKG); otherwise, writes to an ESRI Shapefile.

    Parameters
    ----------
    records : list of dict
        List of records, each containing geometry and attribute data for the layer.
    info : dict
        Configuration dictionary containing output directory, CRS, and layer name (index).

    Returns
    -------
    None
        The function writes files to disk and does not return a value.
    """
    # Use CRS from info['coords']
    # Normalize and assign CRS
    gdf = gpd.GeoDataFrame(records)
    crs_obj = CRS.from_user_input(info["coords"]["crs"])
    auth = crs_obj.to_authority()
    gdf = gdf.set_crs(CRS.from_epsg(int(auth[1])), allow_override=True)

    layer_name = info["index"]
    if GPKG_ACTIVE is True:
        # Write to GeoPackage
        if first_layer:
            gdf.to_file(f"{layer_name}.gpkg", driver="GPKG", layer=layer_name)
            first_layer = False
        else:
            gdf.to_file(f"{layer_name}.gpkg", driver="GPKG", layer=layer_name, mode="a")
    else:
        # Write a shapefile per-percentile for compatibility with CRS
        shp_path = Path(info["output_dir"]) / f"{layer_name}.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
    return

def save_contours(contours, info):
    """
    Save extracted contour geometries to disk as a spatial layer (GeoPackage or Shapefile).

    Converts a nested list of contour arrays (per percentile or horizon time, per time slice)
    into real-world geometries and saves them using the save_layer function. The output format
    is determined by the configuration and the GPKG_ACTIVE flag.

    Parameters
    ----------
    contours : list
        Nested list of contour arrays. For each percentile or horizon time, contains a list of
        per-time-slice contour arrays (each array is an (N, 2) array of pixel indices).
    info : dict
        Configuration dictionary with output path, coordinate arrays, and index-specific keys
        (e.g., 'percentiles' or 'horizon_times').

    Returns
    -------
    None
        The function writes files to disk and does not return a value.
    """    
    if info["index"] == "presence_boundary":
        key_ = "percentiles"
    elif info["index"] == "influence_extent":
        key_ = "horizon_times"

    records = []
    for i_key, key in enumerate(info[info["index"]][key_]):
        for t_idx, slice_contours in enumerate(contours[i_key]):
            for contour in slice_contours:
                geom = obtain_geometry(contour, info)
                if info["index"] == "presence_boundary":
                    records.append({"geometry": geom, "percentile": str(key), "time": info["dates"][t_idx]})
                elif info["index"] == "influence_extent":
                    records.append({"geometry": geom, "horizon_time": str(key)})
    
    save_layer(records, info)

    return


def save_arrays_to_netcdf(arrays, info):
    """
    Save a list of arrays (per percentile) to a NetCDF file with spatial coordinates and metadata.

    This function writes angle and magnitude arrays (per percentile) to a NetCDF file, including
    CF-Convention metadata and CRS (EPSG:25830 or as specified) for compatibility with QGIS/MDAL.
    The output includes coordinate variables, grid mapping, and global attributes for spatial reference.

    Parameters
    ----------
    arrays : list of [angle, magnitude]
        List of [angle, magnitude] arrays for each percentile. Each array should be 2D or 3D (optionally 4D with time).
    info : dict
        Configuration dictionary containing coordinate arrays, percentiles, CRS, and output directory.

    Returns
    -------
    outpath : Path
        Path to the saved NetCDF file.
    """


    # Spatial coordinates
    x = np.asarray(info['coords']['x'])
    y = np.asarray(info['coords']['y'])
    coords = {'y': y, 'x': x}

    # Separate angle and magnitude arrays
    angle_list = [arrs[0] for arrs in arrays]
    magnitude_list = [arrs[1] for arrs in arrays]
    angle_arr = np.stack(angle_list, axis=0)
    magnitude_arr = np.stack(magnitude_list, axis=0)

    # Detect array dimensions to set NetCDF variable dimensions
    if angle_arr.ndim == 4:
        dims = ['percentile', 'time', 'y', 'x']
        time = np.asarray(info['dates'])
        coords['time'] = time
    elif angle_arr.ndim == 3:
        dims = ['percentile', 'y', 'x']
    else:
        raise ValueError('Each array must be 2D or 3D')

    coords['percentile'] = np.array(info["percentiles"], dtype=float)

    # Standard attributes for coordinates (CF conventions)
    x_attrs = {
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "units": "m"
    }
    y_attrs = {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "units": "m"
    }

    # CRS auxiliary variable (CF-1.8): contains projection information for QGIS/MDAL compatibility
    crs_code = info['coords']['crs']
    crs_obj = CRS.from_user_input(crs_code)
    epsg = crs_obj.to_epsg()
    wkt = crs_obj.to_wkt()
    crs_attrs = {
        "grid_mapping_name": "transverse_mercator" if crs_obj.to_dict().get("proj") == "tmerc" else "latitude_longitude",
        "epsg_code": f"EPSG:{epsg}" if epsg else "",
        "crs_wkt": wkt
    }

    # Extra global attributes for QGIS/MDAL compatibility
    proj4 = crs_obj.to_proj4() if hasattr(crs_obj, 'to_proj4') else ""
    global_attrs = {
        "crs": wkt,
        "spatial_ref": wkt,
        "crs_wkt": wkt,
        "proj4": proj4
    }

    # Auxiliary variable spatial_ref (string type, WKT value, with attributes)
    spatial_ref_attrs = {
        "spatial_ref": wkt,
        "crs_wkt": wkt,
        "grid_mapping_name": crs_attrs["grid_mapping_name"],
        "epsg_code": crs_attrs["epsg_code"]
    }

    wkt_bytes = wkt.encode('utf-8')
    strlen = len(wkt_bytes)
    spatial_ref_char = np.frombuffer(wkt_bytes, dtype='S1')

    ds = xr.Dataset(
        {
            'angle': (dims, angle_arr, {"grid_mapping": "spatial_ref"}),
            'magnitude': (dims, magnitude_arr, {"grid_mapping": "spatial_ref"}),
            'spatial_ref': (('strlen',), spatial_ref_char, spatial_ref_attrs)
        },
        coords={
            'x': ('x', x, x_attrs),
            'y': ('y', y, y_attrs),
            'percentile': ('percentile', np.array(info["percentiles"], dtype=float)),
            **({'time': ('time', time)} if angle_arr.ndim == 4 else {}),
            'strlen': np.arange(strlen)
        },
        attrs=global_attrs
    )

    # Guardar
    outpath = Path(info["output_dir"]) / (str(info["index"]) + ".nc")
    ds.to_netcdf(outpath)

    return outpath

def load_results(results_path):
    """
    Load indicator analysis results from a results directory (JSON + NPZ format).

    This function reconstructs the results of a spatiotemporal indicator analysis from a directory
    containing a metadata.json file and an arrays.npz file. It returns a dictionary mapping each
    index name to a list of (contours, mean_map) tuples for each simulation.

    Parameters
    ----------
    results_path : str or Path
        Path to the results directory or directly to the metadata.json file.

    Returns
    -------
    results : dict
        Dictionary containing indicator results for each index.
        Format: {index_name: [(contours, mean_map), ...]}

    Examples
    --------
    >>> from environmentaltools.spatiotemporal import raster
    >>> results = raster.load_results("results/indicator_results_20250112_143022")
    >>> # Access results for specific index
    >>> for contours, mean_map in results['mean_presence_boundary']:
    ...     print(f"Found {len(contours)} contours")
    """
    from pathlib import Path
    
    results_path = Path(results_path)
    
    # Handle both directory and metadata.json file paths
    if results_path.is_file() and results_path.name == "metadata.json":
        results_dir = results_path.parent
        metadata_file = results_path
    else:
        results_dir = results_path
        metadata_file = results_dir / "metadata.json"
    
    arrays_file = results_dir / "arrays.npz"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not arrays_file.exists():
        raise FileNotFoundError(f"Arrays file not found: {arrays_file}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load arrays
    arrays = np.load(arrays_file)
    
    # Reconstruct results dictionary
    results = {}
    for index_name, index_meta in metadata["indices"].items():
        results[index_name] = []
        
        for sim_idx in range(index_meta["n_simulations"]):
            # Load mean_map
            mean_map_key = index_meta["mean_map_keys"][sim_idx]
            mean_map = arrays[mean_map_key]
            
            # Load contours
            contours = []
            for contour_key in index_meta["contour_keys"][sim_idx]:
                contours.append(arrays[contour_key])
            
            results[index_name].append((contours, mean_map))
    
    logger.info(f"Results loaded from: {results_dir}")
    return results