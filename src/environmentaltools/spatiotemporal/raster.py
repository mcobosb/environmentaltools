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

from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd

from environmentaltools.spatiotemporal import indicators
from environmentaltools.spatiotemporal.utils import (
    band,
    calculate_grid_angle_and_create_rotated_mesh,
    refinement,
    save_matrix_to_netcdf,
)
import json
from loguru import logger


def calculate_temporal_differences(info):
    """
    Calculate temporal differences between elevation layers and analyze change statistics.
    
    This function computes the differences between consecutive time steps in elevation
    data and calculates what percentage of cells (x*y*t) are different from zero
    relative to the total number of cells.
    
    Parameters
    ----------
    info : dict
        Configuration dictionary containing datacube filenames and processing parameters.
        
    Returns
    -------
    dict
        Dictionary containing temporal difference statistics:
        - 'total_cells': Total number of cells across all dimensions (x*y*t)
        - 'changed_cells': Number of cells with non-zero differences
        - 'change_percentage': Percentage of cells that changed
        - 'change_statistics': Additional statistics per simulation
        
    Notes
    -----
    The function processes each NetCDF file and calculates:
    1. Temporal differences: elevation[t+1] - elevation[t]
    2. Counts non-zero differences across all space-time dimensions
    3. Reports percentage of changed cells relative to total
    """
    logger.info("Starting temporal difference analysis of elevation layers")
    
    total_cells_all_sims = 0
    changed_cells_all_sims = 0
    change_stats = {}
    
    for sim_idx, datacube_file in enumerate(info["datacube_filenames"]):
        logger.info(f"Processing simulation {sim_idx + 1}: {datacube_file}")
        
        # Load NetCDF data
        with xr.open_dataset(datacube_file) as ds:
            # Assume elevation data is in a variable (adjust variable name as needed)
            # Use the first data variable if elevation/z not found
            var_name = list(ds.data_vars.keys())[0]
            elevation_data = ds[var_name]
            logger.info(f"Using variable '{var_name}' of the data cube")
            
            # Get dimensions
            time_dim = elevation_data.dims[0]  # Usually 'time'
            spatial_dims = elevation_data.dims[1:]  # Usually ['y', 'x'] or ['lat', 'lon']
            
            n_times, n_y, n_x = elevation_data.shape
            total_cells_sim = n_times * n_y * n_x
            
            logger.info(f"Data dimensions: {n_times} times × {n_y} × {n_x} = {total_cells_sim:,} total cells")
            
            # Calculate temporal differences (t+1 - t)
            if n_times < 2:
                logger.warning(f"Simulation {sim_idx + 1} has only {n_times} time step(s). Cannot calculate temporal differences.")
                continue
                
            temporal_diffs = elevation_data.diff(dim=time_dim)
            
            # Count non-zero differences
            # Use a small tolerance to account for floating point precision
            tolerance = 1e-10
            non_zero_mask = np.abs(temporal_diffs) > tolerance
            changed_cells_sim = int(non_zero_mask.sum())
            
            # Total cells for differences (one less time step than original)
            total_diff_cells_sim = (n_times - 1) * n_y * n_x
            
            change_percentage_sim = (changed_cells_sim / total_diff_cells_sim) * 100
            
            # Store statistics for this simulation
            change_stats[f'simulation_{sim_idx + 1}'] = {
                'total_cells': total_cells_sim,
                'total_diff_cells': total_diff_cells_sim,
                'changed_cells': changed_cells_sim,
                'change_percentage': change_percentage_sim,
                'dimensions': {'time': n_times, 'y': n_y, 'x': n_x},
                'mean_absolute_change': float(np.abs(temporal_diffs).mean()),
                'max_absolute_change': float(np.abs(temporal_diffs).max()),
                'std_change': float(temporal_diffs.std())
            }
            
            # Accumulate for overall statistics
            total_cells_all_sims += total_diff_cells_sim
            changed_cells_all_sims += changed_cells_sim
            
            logger.info(f"Simulation {sim_idx + 1}: {changed_cells_sim:,}/{total_diff_cells_sim:,} cells changed ({change_percentage_sim:.2f}%)")
    
    # Calculate overall statistics
    overall_change_percentage = (changed_cells_all_sims / total_cells_all_sims) * 100 if total_cells_all_sims > 0 else 0
    
    results = {
        'total_cells': total_cells_all_sims,
        'changed_cells': changed_cells_all_sims,
        'change_percentage': overall_change_percentage,
        'change_statistics': change_stats,
        'summary': {
            'total_simulations': len(info["datacube_filenames"]),
            'cells_analyzed': total_cells_all_sims,
            'cells_changed': changed_cells_all_sims,
            'percentage_changed': overall_change_percentage
        }
    }
    
    return results


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
        processing settings. Expected keys include:
        
        - "input_dtm" : Path
            Path to the input Digital Terrain Model file
        - "no_sims" : int
            Number of simulations to process
        - "project" : dict
            Dictionary with paths for input/output project
        - "index" : list
            List of indices for processing
        - "region_name" : str
            Name of the region being processed
        - "fld_portion" : int, optional
            Flood portion parameter (default: 3)
        - "seasons" : dict, optional
            Seasonal month definitions

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
    1. Verifies existence of input DTM file
    2. Checks simulation project and catalogs available TIFF files
    3. Sets default values for optional parameters
    4. Defines seasonal month groupings if not provided
    5. Creates auxiliary output directory structure
    6. Validates flood portion parameter

    TODO items:
    - Check that level series files exist
    - Validate that max_level has data for all months and years
    """
    # Check if index exists
    if not "project" in info.keys():
        raise ValueError("Project configuration not specified.")
    
    if "index" not in info["project"].keys():
        logger.error("Index not specified in configuration.")
        raise ValueError("Index not specified in configuration.")
    
    # Check if index is a list, if not convert to list
    if not isinstance(info["project"]["index"], list):
        info["index"] = [info["project"]["index"]]
    else:
        info["index"] = info["project"]["index"]
    
    # Check valid index values
    for index in info["index"]:
        if index not in ["mean_presence_boundary", "maximum_influence_extent", "threshold_exceedance_frequency", "permanently_affected_zone",
            "mean_representative_value", "return_period_based_extreme_value", "spatial_change_rate", "functional_area_loss",
            "critical_boundary_retreat", "neighborhood", "neighborhood_gradient_influence", "environmental_convergence",
            "neighborhood_polarization", "local_persistence", "environmental_risk", "directional_influence",
            "multivariate_neighborhood_synergy", "spatiotemporal_coupling", "multivariate_threshold_exceedance",
            "directional_co_evolution", "multivariate_persistence", "multivariate_recovery"]:
            logger.error(f"Invalid index specified: {index}")
            raise ValueError(f"Invalid index specified: {index}")


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

    # Obtain netcdf files
    info["datacube_filenames"] = [str(file) for file in info["project"]["input_path"].rglob("*.nc")]
    
    # Obtain only the files for the number of simulations
    if len(info["datacube_filenames"]) < info["project"]["no_sims"]:
        raise ValueError(f"Number of NetCDF files ({len(info['datacube_filenames'])}) is lower than the number of simulations ({info['project']['no_sims']}).")
    
    info["datacube_filenames"] = info["datacube_filenames"][:info["project"]["no_sims"]]

    # Obtain level files
    info["level_filenames"] = [str(file) for file in info["project"]["input_path"].rglob("*.csv")]

    # Obtain only the files for the number of simulations
    if len(info["level_filenames"]) < info["project"]["no_sims"]:
        raise ValueError(f"Number of level files ({len(info['level_filenames'])}) is lower than the number of simulations ({info['project']['no_sims']}).")

    info["level_filenames"] = info["level_filenames"][:info["project"]["no_sims"]]

    # Verify date consistency between NetCDF and CSV files
    # This ensures that all dates in the NetCDF time dimension have corresponding data in the CSV
    # It also ensure that variables are in the level files
    for j, filename in enumerate(info["level_filenames"]):
        levels = pd.read_csv(
                        filename,
                        sep=",",
                        index_col=0,
                    )
        
        # Get time metadata from NetCDF file
        with xr.open_dataset(info["datacube_filenames"][j]) as ds:
            netcdf_dates = pd.to_datetime(ds.time.values)
        
        # Convert CSV index to datetime for comparison
        csv_dates = pd.to_datetime(levels.index)
        
        # Check that all NetCDF dates are present in CSV
        missing_dates = []
        for date in netcdf_dates:
            if date not in csv_dates:
                missing_dates.append(date)
        
        if missing_dates:
            raise ValueError(f"NetCDF file {info['datacube_filenames'][j]} contains dates not found in CSV file {filename}: {missing_dates}")
        
        # Optional: Check for extra dates in CSV (informational only)
        extra_dates = []
        for date in csv_dates:
            if date not in netcdf_dates:
                extra_dates.append(date)
        
        if extra_dates:
            logger.warning(f"CSV file {filename} contains {len(extra_dates)} extra dates not in NetCDF: {extra_dates[:5]}{'...' if len(extra_dates) > 5 else ''}")

        # Check that all variables in the level files are present in the NetCDF files
        for variable in info["project"]["variables"]:
            if variable not in levels.columns:
                raise ValueError(f"Variable {variable} from project configuration not found in CSV file {filename}.")
            if variable not in ds.data_vars:
                raise ValueError(f"Variable {variable} from project configuration not found in NetCDF file {info['datacube_filenames'][j]}.")

        logger.info(f"Data verification successful for simulation {j+1}: All {len(netcdf_dates)} dates and variables found in NetCDF and CSV file")

    info["dates"] = csv_dates

    # Check the months for high and low seasons
    if not "seasons" in info:
        info["seasons"] = {
            "AN": "annual",
            "TA": [4, 5, 6, 7, 8, 9],
            "TB": [1, 2, 3, 10, 11, 12],
        }

    # Create result output directory structure
    info["project"]["output_files"] = {}
    for index in info["index"]:
        info["project"]["output_files"][index] = {}

        info["project"]["output_files"][index]["matrix"] = (
            Path(info["project"]["output_path"])
            / index
            / "matrix"
        )

        # Check and create results folders
        for key in info["project"]["output_files"][index]:
            info["project"]["output_files"][index][key].expanduser().mkdir(parents=True, exist_ok=True)


    # Check horizon times are in DTM dates
    if "horizon_times" in info["temporal"].keys():
        for horizon_time in info["temporal"]["horizon_times"]:
            if pd.to_datetime(horizon_time) not in info["dates"]:
                raise ValueError(
                    "Horizon time %s do not match available dates.", horizon_time
                )

    # Check return periods are positive
    if "return_periods" in info:
        for rp in info["return_periods"]:
            if rp <= 0:
                logger.error(f"Return period {rp} is not positive.")
                raise ValueError(f"Return period {rp} is not positive.")

    # TODO: if horizon_times and return_periods are both specified, modify input files accordingly

    # Check mesh size to alert about memory usage
    # Get coordinates metadata from NetCDF file
    with xr.open_dataset(info["datacube_filenames"][j]) as ds:
        x_coords, y_coords = ds.x.values, ds.y.values
    nx, ny = x_coords.size, y_coords.size
    total_points = nx * ny
    if total_points > 10**7:  # 10 million points
        logger.warning(
            f"Very large mesh size: {nx}x{ny} = {total_points:,} points. "
            f"This may cause memory issues. "
            f"Consider increasing grid_size (current: {info['grid_size']}) to reduce resolution."
        )
    elif total_points > 10**6:  # 1 million points
        logger.info(
            f"Considerable mesh size: {nx}x{ny} = {total_points:,} points. "
            f"Monitor memory usage."
        )
    # Check refinement parameter
    if "refinement" in info["parameters"].keys():
        if info["parameters"]["refinement"]:
            if "refinement_size" not in info["parameters"].keys():
                raise ValueError("Refinement size not specified for refinement processing.")
    else:
        info["parameters"]["refinement"] = False

    return


def post_treatment(info):
    """
    Perform post-treatment preprocessing for spatiotemporal raster analysis.

    This function performs preprocessing steps after input validation, including
    temporal difference analysis, level data processing, and refinement band creation.
    It prepares the data structures needed for the main analysis workflow.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing project parameters, file paths, and
        processing settings. Must include:
        - datacube_filenames: List of NetCDF file paths
        - level_filenames: List of CSV file paths with threshold levels  
        - project.variables: List of variable names to process
        - project.no_sims: Number of simulations to process
        - refinement_size: Grid refinement parameter

    Returns
    -------
    dict
        Updated configuration dictionary with additional preprocessing results:
        - band_levels: Array with [min_value, max_value] across all simulations
        - band: Refinement band for spatial processing
        - coords: Dictionary with refined grid coordinates X and Y

    Notes
    -----
    The function performs the following preprocessing steps:
    1. Temporal difference analysis on elevation data
    2. Calculation of absolute min/max values across all simulations
    3. Creation of refinement bands for spatial processing
    4. Generation of refined grid coordinates for interpolation

    The temporal difference analysis provides statistics on how much the elevation
    data changes between consecutive time steps across all simulations.
    """

    # Load initial data cube for refinement band creation
    initial_file = info["datacube_filenames"][0]
    data_cube = xr.open_dataset(initial_file) # [info["project"]["variables"][0]]

    # Perform temporal difference analysis across all simulations
    results = calculate_temporal_differences(info)
    # TODO: log results as needed

    # Initialize lists to store min/max values from all simulations
    min_values = []
    max_values = []

    # Process each simulation to find absolute min/max values
    for sim in range(info["project"]["no_sims"]):
        # Load threshold levels data for current simulation
        data = pd.read_csv(
            info['level_filenames'][sim],
            sep=",",
            index_col=0
        )[info["project"]["variables"]]
    
        # Calculate min and max values from this simulation's data
        sim_min = data.min().min()  # Absolute minimum across all columns and rows
        sim_max = data.max().max()  # Absolute maximum across all columns and rows

        min_values.append(sim_min)
        max_values.append(sim_max)

    # Calculate absolute extremes across all simulations
    absolute_min = min(min_values)
    absolute_max = max(max_values)

    # Create band levels array for refinement processing
    info["band_levels"] = np.array([absolute_min, absolute_max])

    # TODO: define the refinement processing
    # Generate refinement band from data cube and level boundaries
    # info["band"], coords = band(data_cube[info["project"]["variables"][0]], info["band_levels"])

    # Initialize coordinate dictionary for refined grid
    # info["coords"] = {}
    # Create refined grid coordinates X and Y for spatial interpolation
    # info["coords"]["X"], info["coords"]["Y"], _ = calculate_grid_angle_and_create_rotated_mesh(
    #     coords["X"], coords["Y"], info["refinement_size"]
    # )

    return info


def binary_matrix(data_cube, levels, info):
    """
    Create binary mask matrix based on threshold levels for spatiotemporal analysis.

    This function generates a binary mask matrix by comparing elevation data in the
    data cube against corresponding threshold levels for each time step. Values below
    the threshold are marked as True (1), values above as False (0).

    Parameters
    ----------
    data_cube : xarray.DataArray
        The input spatiotemporal data cube containing elevation or other environmental
        variable data with dimensions (time, y, x).
    levels : pandas.DataFrame
        DataFrame containing threshold levels indexed by time/date, with columns
        corresponding to variables. Must have the same temporal index as data_cube.
    info : dict
        Configuration dictionary containing project parameters, including:
        - project.variables: List of variable names to process

    Returns
    -------
    numpy.ndarray
        Binary mask array with the same shape as data_cube, where:
        - 1 (True): Values below the threshold level
        - 0 (False): Values above or equal to the threshold level

    Notes
    -----
    The function performs temporal iteration over the data cube, applying
    thresholds for each time step. This is commonly used for:
    - Flood risk analysis (areas below water level)
    - Environmental threshold exceedance analysis
    - Binary classification of environmental conditions
    
    The mask creation follows the pattern:
    mask[t, y, x] = data_cube[t, y, x] < levels[t, variable]
    """
    # Initialize binary mask array with same shape as data cube
    bin_mask = np.zeros_like(data_cube)
    
    # Create binary mask for each time step based on threshold levels
    for k, date in enumerate(levels.index):
        # Create mask where data values are below the threshold level
        mask = data_cube.sel(time=date) < levels.loc[date, info["project"]["variables"][0]]

        # Store the mask in the binary mask array
        bin_mask[k, :, :] = mask.values

    return bin_mask


def analysis(info=None):
    """
    Execute the complete spatiotemporal raster analysis workflow.

    This is the main analysis function that orchestrates the entire spatiotemporal
    raster processing pipeline. It loads configuration, validates inputs, processes
    each simulation and index combination, and generates binary matrices for
    spatial analysis.

    Parameters
    ----------
    info : dict
        Configuration dictionary containing project parameters, input/output paths,
        and analysis settings.

    Returns
    -------
    dict
        Updated configuration dictionary with processed results and metadata.

    Notes
    -----
    The analysis workflow includes:
    1. Configuration loading and validation
    2. Input data verification and preprocessing
    3. Binary matrix generation for each simulation and index
    4. NetCDF output file creation
    5. Progress logging and error handling

    The function processes multiple simulations and analysis indices as specified
    in the configuration, creating separate output files for each combination.

    Examples
    --------
    >>> from pathlib import Path
    >>> config_file = Path("config/coastal_analysis.json")
    >>> results = analysis(config_file)
    >>> print(f"Processed {len(results['datacube_filenames'])} simulations")
    """
    logger.info("="*60)
    logger.info("STARTING SPATIOTEMPORAL RASTER ANALYSIS")
    logger.info("="*60)
    
    # Validate input data and setup processing parameters
    check_inputs(info)
     
    result = {}
    # Process each analysis index specified in configuration
    for k, index in enumerate(info["project"]["index"]):
        logger.info(f"Starting post-processing for project index {info['project']['index'][k]} ---")
        
        # Initialize result list for this index
        result[index] = []
        
        # Process each simulation for the current index
        for sim_no in range(info["project"]["no_sims"]):
            # Generate output filename for current simulation and index
            output_filename = (
                info["project"]["output_files"][index]["matrix"]
                / f"{index}_sim_{str(sim_no+1).zfill(4)}.nc"
            )

            # Skip processing if output file already exists
            if output_filename.exists():
                logger.info(f"Output file {output_filename} already exists. Skipping simulation {sim_no + 1}.")
                continue
            else:
                logger.info(f"Processing simulation {sim_no + 1} for index {index}.")
                
                # Load spatiotemporal data cube for current simulation
                file_path = info["datacube_filenames"][sim_no]

                # Extract the specified environmental variable from NetCDF file
                data_cube = xr.open_dataset(file_path)[info["project"]["variables"][0]]
                coords = data_cube.coords

                # Load threshold levels from CSV file for current simulation
                levels = pd.read_csv(
                    info['level_filenames'][sim_no],
                    sep=",",
                    index_col=0,
                )

                # Generate binary mask matrix based on threshold comparison
                bin_mask = binary_matrix(data_cube, levels, info)

                # Save binary matrix to NetCDF format with metadata
                save_matrix_to_netcdf(
                    bin_mask,
                    coords,
                    data_cube.time,  # Use time coordinate from data_cube
                    info,
                    sim_no,
                    output_filename,
                )
                logger.info(f"Saving {output_filename}")
        
        # post_treatment(info)
    
        logger.info("COMPUTING INDICATORS FROM BINARY MATRIX")
        logger.info("="*60)


    # Si el usuario especifica percentiles, calcularlos sobre el stack de data_cubes
    if 'percentiles' in info.get('statistics', {}):
        percentiles = info['statistics']['percentiles']
        for idx in info['project']['index']:
            # Usar open_mfdataset para abrir todos los NetCDF de simulaciones a la vez
            nc_paths = [
                str(
                    info["project"]["output_files"][idx]["matrix"]
                    / f"{idx}_sim_{str(sim_no+1).zfill(4)}.nc"
                )
                for sim_no in range(info["project"]["no_sims"])
            ]
            data_cubes_stack = xr.open_mfdataset(nc_paths, concat_dim='simulation', combine='nested')
            indicator_func = getattr(indicators, idx)
            percentiles_result = {}
            for p in percentiles:
                # Calcular el indicador para el percentil p sobre la dimensión 'simulation'
                data_cube_percentil = data_cubes_stack.quantile(int(p)/100, dim='simulation')
                indicator_result = indicator_func(data_cube_percentil)
                percentiles_result[f'p{int(p)}'] = indicator_result
            result[idx + '_percentiles'] = percentiles_result

    # Save results to pickle file
    save_results(result, info)
    
    return info, result


def save_results(results, info):
    """
    Save indicator analysis results to JSON + NPZ format.
    
    Saves results in a portable format using JSON for metadata and numpy's
    .npz format for array data. This avoids pickle compatibility issues.
    
    Parameters
    ----------
    results : dict
        Dictionary containing indicator results for each index.
        Format: {index_name: [(contours, mean_map), ...]}
    info : dict
        Configuration dictionary containing output path information.
    
    Returns
    -------
    Path
        Path to the saved results directory.
    """
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    output_path = Path(info["project"]["output_path"])
    results_dir = output_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir / f"indicator_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata and arrays
    metadata = {"indices": {}}
    arrays_to_save = {}
    
    for index_name, simulations in results.items():
        metadata["indices"][index_name] = {
            "n_simulations": len(simulations),
            "contour_keys": [],
            "mean_map_keys": []
        }
        
        for sim_idx, (contours, mean_map) in enumerate(simulations):
            # Save mean_map
            mean_map_key = f"{index_name}_sim{sim_idx}_mean_map"
            arrays_to_save[mean_map_key] = mean_map
            metadata["indices"][index_name]["mean_map_keys"].append(mean_map_key)
            
            # Save contours (each contour separately)
            metadata["indices"][index_name]["contour_keys"].append([])
            for contour_idx, contour in enumerate(contours):
                contour_key = f"{index_name}_sim{sim_idx}_contour{contour_idx}"
                arrays_to_save[contour_key] = contour
                metadata["indices"][index_name]["contour_keys"][-1].append(contour_key)
    
    # Save metadata as JSON
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save arrays as NPZ
    arrays_file = output_dir / "arrays.npz"
    np.savez_compressed(arrays_file, **arrays_to_save)
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Metadata: {metadata_file}")
    logger.info(f"  - Arrays: {arrays_file}")
    return output_dir


def load_results(results_path):
    """
    Load indicator analysis results from JSON + NPZ format.
    
    Parameters
    ----------
    results_path : str or Path
        Path to the results directory (or metadata.json file).
    
    Returns
    -------
    dict
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



#                 # Reading DEM
#                 data_cube = xr.open_dataset(file_path)
#                 if info["refinement"]:
#                     # Refinement
#                     Z = refinement(data_cube, info["band"], coords)
#                 else:
#                     Z = data_cube["z"]

#                 # Mask of data below level
#                 mask = Z < level

#                 # Include mask into bin_mask - simulations and year
#                 bin_mask[sim_no][date] = mask

    
#             # if info["refinement"]:
#         #     # Refinement
#         #     Z = refinement(data_cube, info["band"], coords)
#         # else:
#         #     Z = data_cube[]