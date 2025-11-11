"""CORDEX climate data download utilities.

This module provides functions to download and process CORDEX (Coordinated Regional
Climate Downscaling Experiment) data from ESGF (Earth System Grid Federation) servers.
"""

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set HOME environment variable for cross-platform compatibility
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser("~")

# Suppress PyESGF facets warning for better user experience
os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = '1'

import pandas as pd
import requests
import xarray as xr

# Dependencies for CORDEX download
from cdo import Cdo
from configobj import ConfigObj

# Modern ESGF access with intake-esgf
try:
    import intake_esgf
    HAS_INTAKE_ESGF = True
except ImportError:
    intake_esgf = None
    HAS_INTAKE_ESGF = False
    logger.warning(
        "⚠️  intake-esgf not found. Using fallback functionality. "
        "Install with: pip install intake-esgf"
    )

# Legacy ESGF access with PyESGF (for CORDEX support)
try:
    from pyesgf.search import SearchConnection
    try:
        from pyesgf.logon import LogonManager
    except ImportError:
        LogonManager = None
    HAS_PYESGF = True
except ImportError:
    SearchConnection = None
    LogonManager = None  
    HAS_PYESGF = False

from tqdm import tqdm
from werkzeug.utils import secure_filename
from loguru import logger

from environmentaltools.common import utils

# ESGF directory for certificates
ESGF_DIR = os.path.expanduser("~/.esg")

# Validate ESGF directory exists when needed
def _validate_esgf_dir() -> None:
    """Validate that ESGF directory and certificates exist.
    
    Raises:
        FileNotFoundError: If ESGF directory or required certificates don't exist.
    """
    if not os.path.exists(ESGF_DIR):
        raise FileNotFoundError(
            f"ESGF directory not found: {ESGF_DIR}\n"
            "Please ensure you have ESGF certificates installed. "
            "You can obtain them by:\n"
            "1. Registering at an ESGF node (e.g., https://esg-dn1.nsc.liu.se/)\n"
            "2. Running bootstrap authentication with your credentials"
        )
    
    cert_file = os.path.join(ESGF_DIR, "credentials.pem")
    cert_dir = os.path.join(ESGF_DIR, "certificates")
    
    if not os.path.exists(cert_file):
        raise FileNotFoundError(
            f"ESGF credentials file not found: {cert_file}\n"
            "Please bootstrap your ESGF certificates first."
        )
    
    if not os.path.exists(cert_dir):
        raise FileNotFoundError(
            f"ESGF certificates directory not found: {cert_dir}\n"
            "Please bootstrap your ESGF certificates first."
        )



def parse_wget_script_to_queries(file_name: str, output_path: str = "") -> dict:
    """Parse ESGF wget script and extract query parameters.

    Reads an ESGF-generated wget script file and extracts CORDEX dataset metadata
    to create structured query dictionaries for data download.

    Args:
        file_name (str): Path to the wget script file from ESGF.
        output_path (str, optional): Directory path where downloaded files will be saved.
            Defaults to empty string (current directory).

    Returns:
        dict: Dictionary mapping indices to query configurations. Each entry contains:
            - filename (str): Target file path for the downloaded data
            - query (dict): CORDEX query parameters including project, variable,
              time_frequency, domain, experiment, ensemble, rcm_version,
              driving_model, and institute

    Example:
        >>> queries = parse_wget_script_to_queries('wget_script.sh', './data')
        >>> print(queries[0]['query']['variable'])
        'tas'
    """
    # Read and parse wget script file
    with open(file_name, "r") as file:
        data = file.read()
        # Extract the section after the header
        data = data.split("EOF--dataset.file.url.chksum_type.chksum")[1]
        data = data.split("\n")[1:-1]

    # Parse filenames from wget URLs
    data = [line.split(" ")[0].replace("'", "").split("_")[:-1] for line in data]
    separator = "_"
    data = [separator.join(line) for line in data]
    # Remove duplicates while preserving order
    filenames = list(dict.fromkeys(data))

    # Extract metadata from filenames using CORDEX naming convention
    # Format: variable_domain_model_experiment_ensemble_rcm_downscaling_frequency
    variables = [line.split("_")[0] for line in filenames]
    domains = [line.split("_")[1] for line in filenames]
    models = [line.split("_")[2] for line in filenames]
    experiments = [line.split("_")[3] for line in filenames]
    ensembles = [line.split("_")[4] for line in filenames]
    rcms = [line.split("_")[5] for line in filenames]
    downscaling_methods = [line.split("_")[6] for line in filenames]
    frequencies = [line.split("_")[7] for line in filenames]

    # Build query dictionary
    queries = {}
    for idx, filename in enumerate(filenames):
        queries[idx] = {
            "filename": os.path.join(output_path, filename) if output_path else filename,
            "query": {
                "project": "CORDEX",
                "variable": variables[idx],
                "time_frequency": frequencies[idx],
                "domain": domains[idx],
                "experiment": experiments[idx],
                "ensemble": ensembles[idx],
                "rcm_version": downscaling_methods[idx],
                "driving_model": models[idx],
                "institute": rcms[idx].split("-")[0],
            },
        }
    return queries


async def download_cordex_dataset(
    query: dict,
    credentials: tuple[str, str],
    point: dict | None = None,
    region: dict | None = None,
) -> None:
    """Download a single CORDEX dataset asynchronously.

    Queries an ESGF server to download CORDEX climate data for a specific point
    or region. Uses PyDAP protocol for efficient data access.

    Args:
        query (dict): Query configuration containing:
            - filename (str): Output file path
            - query (dict): CORDEX parameters (project, variable, domain, etc.)
        credentials (tuple[str, str]): Authentication credentials as (openid, password)
            for ESGF server access. Should be loaded from secure config, never hardcoded.
        point (dict, optional): Geographic point with 'lat' and 'lon' keys
            for single-point extraction. Defaults to None.
        region (dict, optional): Bounding box with 'lat' and 'lon' arrays
            ([lat_min, lat_max], [lon_min, lon_max]) for regional extraction.
            Defaults to None.

    Returns:
        None: Data is saved directly to NetCDF file.

    Note:
        Requires OpenDAP access to ESGF servers. One of point or region must be
        specified, but not both.
    """
    try:
        # Query ESGF server and download dataset
        dataset, _ = utils.cordex(
            query["query"],
            openid=credentials[0],
            password=credentials[1],
            pydap=True,
            bootstrap=True,
        )
        
        # Small delay to avoid overwhelming the server
        await asyncio.sleep(1)
        
        # Extract data based on spatial selection
        if point is not None:
            # Extract single point data
            extracted_data = utils.xrnearest(dataset, point["lat"], point["lon"])
            output_filename = f"{query['filename']}_{point['lat']}_{point['lon']}.nc"
            extracted_data.to_netcdf(output_filename)
            logger.info(f"Downloaded point data: {output_filename}")
            
        elif region is not None:
            # Extract regional data
            extracted_data = utils.subregion(dataset, region["lat"], region["lon"])
            output_filename = (
                f"{query['filename']}_"
                f"{region['lat'][0]}_{region['lat'][1]}_"
                f"{region['lon'][0]}_{region['lon'][1]}.nc"
            )
            extracted_data.to_netcdf(output_filename)
            logger.info(f"Downloaded regional data: {output_filename}")
            
    except Exception as e:
        logger.error(f"Download failed for {query['filename']}: {str(e)}")
        
    return


def download_cordex_data(
    wget_script_file: str,
    credentials: tuple[str, str],
    output_path: str = "",
    point: pd.DataFrame | None = None,
    region: pd.DataFrame | None = None,
) -> None:
    """Download CORDEX climate data from ESGF servers.

    Orchestrates concurrent downloads of multiple CORDEX datasets based on queries
    extracted from an ESGF wget script. Supports both point-based and region-based
    data extraction.

    Args:
        wget_script_file (str): Path to the ESGF-generated wget script file containing
            download URLs and dataset information.
        credentials (tuple[str, str]): ESGF authentication credentials as
            (openid_url, password). Example:
            ("https://esg-dn1.nsc.liu.se/esgf-idp/openid/username", "password")
        output_path (str, optional): Directory where files will be saved.
            Directory is created if it doesn't exist. Defaults to current directory.
        point (pd.DataFrame, optional): DataFrame with 'lat' and 'lon' columns
            specifying geographic points for data extraction. Each row represents
            one point location. Defaults to None.
        region (pd.DataFrame, optional): DataFrame with 'lat' and 'lon' columns
            containing [min, max] arrays defining bounding boxes. Each row represents
            one region. Defaults to None.

    Returns:
        None: Downloads data and saves to NetCDF files in the specified output path.

    Raises:
        ValueError: If neither point nor region is specified, or if both are specified.

    Example:
        >>> import pandas as pd
        >>> credentials = ("https://esg-dn1.nsc.liu.se/esgf-idp/openid/user", "pass")
        >>> points = pd.DataFrame({'lat': [40.0, 41.0], 'lon': [-3.0, -2.0]})
        >>> download_cordex_data('wget_script.sh', credentials, './data', point=points)

    Note:
        - Uses asyncio for concurrent downloads to improve efficiency
        - One and only one of point or region must be provided
        - Large downloads may take significant time depending on data volume
    """
    # Create output directory if specified
    os.makedirs(output_path, exist_ok=True)

    # Parse wget script to extract query configurations
    queries = parse_wget_script_to_queries(wget_script_file, output_path)

    # Set up asyncio event loop for concurrent downloads
    loop = asyncio.get_event_loop()
    tasks = []
    
    if point is not None:
        # Create download tasks for each point and each query
        point_dict = {}
        for coord in point.itertuples():
            point_dict["lat"] = coord.lat
            point_dict["lon"] = coord.lon

            for query_idx in queries:
                logger.info(f"Queuing download: {queries[query_idx]['filename']}")
                tasks.append(
                    download_cordex_dataset(
                        queries[query_idx], credentials, point=point_dict
                    )
                )

    elif region is not None:
        # Create download tasks for each region and each query
        region_dict = {}
        for coord in region.itertuples():
            region_dict["lat"] = coord.lat
            region_dict["lon"] = coord.lon

            for query_idx in queries:
                logger.info(f"Queuing download: {queries[query_idx]['filename']}")
                tasks.append(
                    download_cordex_dataset(
                        queries[query_idx], credentials, region=region_dict
                    )
                )
    else:
        raise ValueError(
            "Neither point nor region specified. One is required for data extraction."
        )

    # Execute all download tasks concurrently
    loop.run_until_complete(asyncio.wait(tasks))
    logger.info("All downloads completed")
    
    return


def download_with_config(
    output_folder: str,
    bootstrap: bool = False,
    pydap: bool = False,
) -> None:
    """Batch download CORDEX data based on configuration files.

    This function provides a workflow for querying and downloading CORDEX data
    using configuration files that specify coordinates and query selections.

    Args:
        query (dict): Dictionary of CORDEX query parameters. Common keys include:
            "project", "domain", "experiment", "time_frequency", and "variable".
        output_folder (str): Directory path where downloaded files and queries
            will be stored.
        download (bool, optional): If True, downloads selected data. If False,
            only generates query file without downloading. Defaults to True.
        bootstrap (bool, optional): If True, generates or renews ESGF certificates
            for authentication. Use when starting fresh or if certificates expired.
            Defaults to False.
        pydap (bool, optional): If True, uses PyDAP data sources. If False, uses
            standard OpenDAP protocol. Defaults to False.

    Note:
        - Expects a ~/.esgf/config.ini file with ESGF credentials in format:
          [credentials]
          openid = your_esgf_openid
          password = your_password
        - Requires CSV files: {output_folder}_coordenadas.csv (coordinates)
          and {output_folder}_seleccion.csv (query selection)
        - Creates output structure: output_folder/coord_XX/files.nc

    Example:
        >>> # Generate queries only
        >>> batch_download_with_config('./data', download=False)
        >>> # Download data after selecting queries
        >>> batch_download_with_config('./data', download=True)
    """
    # Create output folder structure
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    query_file = output_path / "queries.xlsx"

    # Load configuration and data files
    coord_file = output_path / "coordinates.csv"
    selection_file = output_path / "selection.csv"

    # Read ESGF credentials from config file
    config_path = os.path.expanduser("~/.esgf/config.ini")
    if not os.path.exists(config_path):
        logger.error(f"ESGF config file not found at {config_path}")
        logger.error("Please create the ~/.esgf/config.ini file with your ESGF credentials")
        logger.error("Format: [credentials]\\nopenid = your_esgf_openid\\npassword = your_password")
        raise FileNotFoundError(f"ESGF config file not found: {config_path}")
    
    config = ConfigObj(config_path)
    
    try:
        openid = config["credentials"]["openid"]
        password = config["credentials"]["password"]
    except KeyError as e:
        logger.error(f"Missing required credential in config file: {e}")
        logger.error("Config file format should be: [credentials]\\nopenid = your_esgf_openid\\npassword = your_password")
        raise
    
    logger.info("📡 Starting CORDEX data download...")
    logger.info("⚠️  Note: ESGF servers may be slow or temporarily unavailable")
    logger.info("🔄 The process includes automatic retry logic for failed connections")
    logger.info("🌍 Multiple ESGF servers will be tried automatically")
    logger.info(f"🔐 Using ESGF credentials from: {config_path}")
    logger.info("💡 Many CORDEX datasets are public - authentication is optional")

    # Load coordinates and query selection
    coords = pd.read_csv(coord_file)[3:]  # Skip first 3 header rows
    all_queries = pd.read_excel(f"{query_file}")
    selection = pd.read_csv(selection_file, header=None).values[:, 0].tolist()

    # Filter queries based on selection
    queries = filter_esgf_queries(selection, all_queries)

    # Download data for each coordinate point
    for coord in coords.itertuples():
        # Create subfolder for each coordinate
        coord_folder = f"coord_{coord.Index:02}"
        Path(f"{output_folder}/{coord_folder}").mkdir(exist_ok=True)

        lat = coord.lat
        lon = coord.lon
        logger.info(f"Processing coordinate: lat={lat}, lon={lon}")

        # Download each selected query for this coordinate
        for q in queries.itertuples():
            filename = (
                f"./{output_folder}/{coord_folder}/"
                f"c{coord.Index:02}-q{q.Index:02}-{secure_filename(q.id)}.nc"
            )

            # Skip if file already exists
            if Path(filename).exists():
                logger.info(f"{q.id} already downloaded")
            else:
                logger.info(f"Downloading {q.id}")

                try:
                    # Download dataset from ESGF
                    dataset, _ = download_esgf_dataset(
                        q.id,
                        openid=openid,
                        password=password,
                        bootstrap=bootstrap,
                        pydap=pydap,
                    )

                    # Extract point of interest and save to NetCDF
                    poi = utils.nearest(dataset, lat, lon)
                    poi.to_netcdf(filename)
                    poi.close()
                    dataset.close()
                    
                except (requests.exceptions.ConnectTimeout, 
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError) as e:
                    logger.error(f"❌ Failed to download {q.id}: {type(e).__name__}: {e}")
                    logger.error("Skipping this dataset and continuing with next one...")
                    continue
                    
                except Exception as e:
                    logger.error(f"❌ Unexpected error downloading {q.id}: {type(e).__name__}: {e}")
                    logger.error("Skipping this dataset and continuing with next one...")
                    continue
       

    logger.info("Finished!")


def query_esgf_catalog(
    query: dict,
    search_url: str = "https://esg-dn1.nsc.liu.se/esg-search",
    distrib: bool = True,
    facets: str | None = None,
) -> pd.DataFrame:
    """Query ESGF catalog and retrieve available datasets.

    Searches the ESGF (Earth System Grid Federation) catalog for datasets
    matching the specified query parameters.

    Args:
        query (dict): Dictionary of CORDEX query parameters. Common keys include:
            - project (str): e.g., "CORDEX"
            - domain (str): e.g., "EUR-11"
            - experiment (str): e.g., "rcp85"
            - time_frequency (str): e.g., "3hr", "day", "mon"
            - variable (list): e.g., ["pr", "tas"]
        search_url (str, optional): ESGF search node URL. Defaults to
            "https://esg-dn1.nsc.liu.se/esg-search".
        distrib (bool, optional): If True, searches across distributed nodes.
            Defaults to True.
        facets (str, optional): Comma-separated list of facets to include in search.
            If None, uses CORDEX-specific defaults. Examples: 'project,domain,variable'
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing metadata for all matching datasets.
            Each row represents one dataset with its full metadata.

    Example:
        >>> query = {"project": "CORDEX", "domain": "EUR-11", "variable": ["pr"]}
        >>> datasets = query_esgf_catalog(query)
        >>> print(f"Found {len(datasets)} datasets")
    """
def query_esgf_catalog(
    query: dict,
    search_url: str = "https://esg-dn1.nsc.liu.se/esg-search",
    distrib: bool = True,
    facets: str | None = None,
) -> pd.DataFrame:
    """Query ESGF catalog and retrieve available datasets using intake-esgf.

    Searches the ESGF (Earth System Grid Federation) catalog for datasets
    matching the specified query parameters using the modern intake-esgf package.

    Args:
        query (dict): Dictionary of CORDEX query parameters. Common keys include:
            - project (str): e.g., "CORDEX"
            - domain (str): e.g., "EUR-11"
            - experiment (str): e.g., "rcp85"
            - time_frequency (str): e.g., "3hr", "day", "mon"
            - variable (list): e.g., ["pr", "tas"]
        search_url (str, optional): Primary ESGF search node URL. Function will
            try alternative nodes if this fails. Defaults to
            "https://esg-dn1.nsc.liu.se/esg-search".
        distrib (bool, optional): If True, searches across distributed nodes.
            Defaults to True.
        facets (str, optional): Comma-separated list of facets to include in search.
            If None, uses CORDEX-specific defaults. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing metadata for all matching datasets.
            Each row represents one dataset with its full metadata.

    Raises:
        ImportError: If intake-esgf is not installed
        ConnectionError: If unable to connect to any ESGF node

    Example:
        >>> query = {"project": "CORDEX", "domain": "EUR-11", "variable": ["pr"]}
        >>> datasets = query_esgf_catalog(query)
        >>> print(f"Found {len(datasets)} datasets")
    """
    if not HAS_INTAKE_ESGF:
        raise ImportError(
            "intake-esgf is required for ESGF operations. "
            "Install with: pip install intake-esgf"
        )

    # List of ESGF indices to try
    indices = [
        "https://esgf-node.llnl.gov/esg-search",
        "https://esgf-index1.ceda.ac.uk/esg-search", 
        "https://esgf.nci.org.au/esg-search",
        search_url  # User-specified URL last
    ]

    results = []
    successful_indices = []

    # Try each index until we get results
    for idx_url in indices:
        try:
            logger.info(f"🔍 Searching ESGF index: {idx_url}")
            
            # Create intake-esgf catalog
            cat = intake_esgf.ESGFCatalog()
            
            # Perform search with intake-esgf
            search_results = cat.search(**query)
            
            if hasattr(search_results, '__len__') and len(search_results) > 0:
                logger.info(f"✅ Found {len(search_results)} results")
                
                # Convert to DataFrame format
                if hasattr(search_results, 'df'):
                    results.extend(search_results.df.to_dict('records'))
                elif hasattr(search_results, 'to_dict'):
                    results.extend([search_results.to_dict()])
                else:
                    # Handle different result formats
                    for item in search_results:
                        if hasattr(item, 'to_dict'):
                            results.append(item.to_dict())
                        else:
                            results.append({"id": str(item), "source": idx_url})
                
                successful_indices.append(idx_url)
                break  # Use first successful index
            else:
                logger.warning(f"No results found at {idx_url}")
                
        except Exception as e:
            logger.warning(f"❌ Failed to search {idx_url}: {type(e).__name__}: {e}")
            continue

    if not results:
        logger.error("No datasets found - possible causes:")
        logger.error("1. Query parameters too restrictive")
        logger.error("2. Network connectivity issues")
        logger.error("3. ESGF servers temporarily unavailable")
        logger.error("4. intake-esgf configuration issues")
        
        # Return empty DataFrame instead of raising error
        logger.warning("Returning empty DataFrame")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(results)
    logger.info(f"📊 Total datasets found: {len(df)} from {len(successful_indices)} indices")
    
    return df


def query_esgf_catalog_pyesgf(
    query: Dict[str, Union[str, List[str]]],
    indices: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Query ESGF catalog using PyESGF (esgf-pyclient) for CORDEX projects.
    
    This function uses the legacy PyESGF client which has proven stable support
    for CORDEX data discovery, unlike intake-esgf which doesn't support CORDEX yet.
    
    Args:
        query: Dictionary of CORDEX query parameters. Common keys include:
            - project (str): e.g., "CORDEX"
            - domain (str): e.g., "EUR-11"
            - experiment (str): e.g., "rcp85"
            - time_frequency (str): e.g., "3hr", "day", "mon"
            - variable (list): e.g., ["pr", "tas"]
        indices: List of ESGF indices to search
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        pd.DataFrame: DataFrame containing metadata for matching datasets
        
    Raises:
        ImportError: If PyESGF is not installed
        ConnectionError: If unable to connect to any ESGF index
    """
    if not HAS_PYESGF:
        raise ImportError(
            "PyESGF (esgf-pyclient) is required for CORDEX projects. "
            "Install with: pip install esgf-pyclient"
        )
    
    # Default ESGF indices for CORDEX
    if indices is None:
        indices = [
            "https://esgf.nci.org.au/esg-search",
            "https://esgf-node.llnl.gov/esg-search",
            "https://esgf-index1.ceda.ac.uk/esg-search", 
            "https://esg-dn1.nsc.liu.se/esg-search"
        ]
    
    results = []
    successful_indices = []
    
    # Try each index until we get results
    for idx_url in indices:
        try:
            logger.info(f"🔍 Searching ESGF index: {idx_url}")
            
            # Connect to ESGF using PyESGF
            conn = SearchConnection(idx_url, distrib=True)
            
            # Use specific facets for CORDEX to avoid warning and improve search efficiency
            # Based on CORDEX-specific metadata structure
            cordex_facets = 'project,domain,variable,experiment,time_frequency,driving_model,institute,rcm_name,ensemble'
            ctx = conn.new_context(facets=cordex_facets)
            
            # Add query parameters to context using PyESGF's constraint system
            applied_filters = []
            for key, value in query.items():
                if isinstance(value, list):
                    # For lists, try each value or join with comma
                    if len(value) == 1:
                        ctx = ctx.constrain(**{key: value[0]})
                        applied_filters.append(f"{key}={value[0]}")
                    else:
                        # Multiple values - use comma-separated
                        filter_value = ','.join(value)
                        ctx = ctx.constrain(**{key: filter_value})
                        applied_filters.append(f"{key}={filter_value}")
                else:
                    ctx = ctx.constrain(**{key: str(value)})
                    applied_filters.append(f"{key}={value}")
            
            logger.info(f"🔍 Applied filters: {', '.join(applied_filters)}")
            
            # Get hit count
            hit_count = ctx.hit_count
            
            if hit_count == 0:
                logger.warning(f"No datasets found at {idx_url}")
                continue
            
            # Limit results for performance and practical use
            max_results = kwargs.get('max_results', 100)  # Default limit of 100 datasets
            actual_results = min(hit_count, max_results)
                
            logger.info(f"✅ Found {hit_count} datasets at {idx_url}, retrieving first {actual_results}")
            successful_indices.append(idx_url)
            
            # Get limited results
            result_count = 0
            for result in ctx.search():
                if result_count >= max_results:
                    logger.info(f"⚡ Reached maximum result limit ({max_results})")
                    break
                dataset_info = {
                    'dataset_id': result.dataset_id,
                    'title': getattr(result, 'title', ''),
                    'source': idx_url,
                    'number_of_files': getattr(result, 'number_of_files', 0),
                    'size': getattr(result, 'size', 0),
                    'version': getattr(result, 'version', ''),
                    'instance_id': getattr(result, 'instance_id', ''),
                    'master_id': getattr(result, 'master_id', ''),
                    # Try different ways to extract metadata
                    'project': getattr(result, 'project', query.get('project', '')),
                    'experiment': getattr(result, 'experiment', query.get('experiment', [''])[0] if isinstance(query.get('experiment'), list) else query.get('experiment', '')),
                    'time_frequency': getattr(result, 'time_frequency', query.get('time_frequency', [''])[0] if isinstance(query.get('time_frequency'), list) else query.get('time_frequency', '')),
                    'variable': getattr(result, 'variable', query.get('variable', [''])[0] if isinstance(query.get('variable'), list) else query.get('variable', '')),
                    'domain': getattr(result, 'domain', query.get('domain', '')),
                    'institute': getattr(result, 'institute', ''),
                    'model': getattr(result, 'model', ''),
                    'driving_model': getattr(result, 'driving_model', ''),
                    'ensemble': getattr(result, 'ensemble', ''),
                    # Ensure we have an 'id' field for compatibility
                    'id': result.dataset_id
                }
                
                # If metadata is still empty, try to parse from dataset_id
                if not dataset_info['domain'] and 'EUR-11' in result.dataset_id:
                    dataset_info['domain'] = 'EUR-11'
                if not dataset_info['experiment'] and 'rcp85' in result.dataset_id:
                    dataset_info['experiment'] = 'rcp85'
                if not dataset_info['variable'] and 'pr' in result.dataset_id:
                    dataset_info['variable'] = 'pr'
                if not dataset_info['project'] and 'cordex' in result.dataset_id.lower():
                    dataset_info['project'] = 'CORDEX'
                results.append(dataset_info)
                result_count += 1
                
            # Break after first successful search to avoid duplicates
            if results:
                break
                
        except Exception as e:
            logger.warning(f"❌ Failed to search {idx_url}: {type(e).__name__}: {e}")
            continue
    
    if not results:
        logger.error("No datasets found - possible causes:")
        logger.error("1. Query parameters too restrictive")
        logger.error("2. Network connectivity issues") 
        logger.error("3. ESGF servers temporarily unavailable")
        logger.warning("Returning empty DataFrame")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    logger.info(f"📊 Found {len(df)} CORDEX datasets using PyESGF")
    
    return df


def download_esgf_dataset_pyesgf(
    dataset_metadata: Dict,
    output_folder: str,
    file_filter: Optional[str] = None,
    **kwargs
) -> List[str]:
    """Download ESGF dataset using PyESGF with metadata dictionary input.
    
    Adapter function that converts dataset metadata dictionary to the format
    expected by the original download_esgf_dataset function.
    
    Args:
        dataset_metadata: Dictionary containing dataset metadata including 'dataset_id' or 'id'
        output_folder: Directory to save downloaded files
        file_filter: Optional filter for specific files (e.g., "*.nc")
        **kwargs: Additional arguments passed to download function
        
    Returns:
        List[str]: List of downloaded file paths
        
    Raises:
        ValueError: If dataset_metadata doesn't contain required ID field
        ImportError: If PyESGF is not available
    """
    if not HAS_PYESGF:
        raise ImportError(
            "PyESGF (esgf-pyclient) is required for CORDEX downloads. "
            "Install with: pip install esgf-pyclient"
        )
    
    # Extract dataset ID from metadata
    dataset_id = dataset_metadata.get('dataset_id') or dataset_metadata.get('id')
    if not dataset_id:
        raise ValueError(
            "Dataset metadata must contain 'dataset_id' or 'id' field. "
            f"Available fields: {list(dataset_metadata.keys())}"
        )
    
    logger.info(f"🌐 Downloading dataset: {dataset_id}")
    logger.info(f"📁 Output folder: {output_folder}")
    
    try:
        # Call the original download function with the dataset ID
        dataset, opendap_urls = download_esgf_dataset(
            query=dataset_id,
            **kwargs
        )
        
        # Convert xarray dataset(s) to file paths
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        if dataset is not None:
            if isinstance(dataset, list):
                # Multiple datasets
                for i, ds in enumerate(dataset):
                    filename = f"{dataset_id.replace('|', '_').replace('/', '_')}_part{i}.nc"
                    if file_filter and not filename.endswith(file_filter.replace('*', '')):
                        continue
                    filepath = output_path / filename
                    ds.to_netcdf(filepath)
                    downloaded_files.append(str(filepath))
                    logger.info(f"💾 Saved: {filepath.name}")
            else:
                # Single dataset
                filename = f"{dataset_id.replace('|', '_').replace('/', '_')}.nc"
                if not file_filter or filename.endswith(file_filter.replace('*', '')):
                    filepath = output_path / filename
                    dataset.to_netcdf(filepath)
                    downloaded_files.append(str(filepath))
                    logger.info(f"💾 Saved: {filepath.name}")
        
        if opendap_urls:
            logger.info(f"🔗 OpenDAP URLs available: {len(opendap_urls)}")
        
        return downloaded_files
        
    except Exception as e:
        logger.error(f"❌ Download failed for {dataset_id}: {e}")
        raise


def filter_esgf_queries(selection: list, queries: pd.DataFrame) -> pd.DataFrame:
    """Filter ESGF query results based on selection indices.

    Args:
        selection (list): List of row indices to select from queries DataFrame.
        queries (pd.DataFrame): Full DataFrame of ESGF query results.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only selected queries
            with their 'id' column.

    Example:
        >>> selection = [0, 5, 10]
        >>> filtered = filter_esgf_queries(selection, all_queries)
    """
    return queries.loc[selection, ["id"]]


def download_esgf_dataset(
    query: str,
    search_url: str = "https://esg-dn1.nsc.liu.se/esg-search",
    distrib: bool = True,
    split_by_variable: bool = False,
    openid: str | None = None,
    password: str | None = None,
    bootstrap: bool = False,
    pydap: bool = False,
) -> tuple[xr.Dataset | list[xr.Dataset], list[str]]:
    """Download ESGF dataset using OpenDAP protocol.

    Downloads climate data from ESGF servers using either PyDAP or standard
    OpenDAP protocol. Supports authentication and multiple file handling.
    Includes automatic fallback to alternative ESGF servers for reliability.
    Authentication is optional - many CORDEX datasets are publicly accessible.

    Args:
        query (str): ESGF dataset ID to download.
        search_url (str, optional): Primary ESGF search node URL. The function
            will automatically try alternative servers if this one fails.
            Defaults to "https://esg-dn1.nsc.liu.se/esg-search".
        distrib (bool, optional): If True, searches across distributed nodes.
            Defaults to True.
        split_by_variable (bool, optional): If True, returns separate datasets
            for each variable. Defaults to False.
        openid (str, optional): ESGF OpenID URL for authentication. If authentication
            fails, the function will continue without credentials. Defaults to None.
        password (str, optional): Password for ESGF authentication. Defaults to None.
        bootstrap (bool, optional): If True, generates or renews certificates.
            Defaults to False.
        pydap (bool, optional): If True, uses PyDAP for data access. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - dataset (xr.Dataset or list[xr.Dataset]): Downloaded dataset(s)
            - opendap_urls (list[str]): List of OpenDAP URLs accessed

    Example:
        >>> dataset, urls = download_esgf_dataset(
        ...     "cordex.output.EUR-11.SMHI.ICHEC-EC-EARTH...",
        ...     openid="https://esg-dn1.nsc.liu.se/esgf-idp/openid/user",
        ...     password="password"
        ... )
    """
    session = None
    ds = None
    opendap_urls = []
    multiple_opendap_urls = []

    # Set up authentication if credentials provided
    if openid and password:
        try:
            if pydap:
                # session = setup_session(openid, password, check_url=openid)
                session = None  # TODO: Implement session setup
            else:
                lm = LogonManager()
                logger.info("🔐 Attempting ESGF authentication...")
                lm.logon_with_openid(openid=openid, password=password, bootstrap=bootstrap)
                logger.info("✅ ESGF authentication successful")
        except Exception as e:
            logger.warning(f"⚠️  Authentication failed: {e}")
            logger.warning("🔓 Continuing without authentication (many CORDEX datasets are public)")
            logger.warning("Some datasets may not be accessible without proper authentication")
            # Continue without authentication - many datasets are public

    # List of alternative ESGF search nodes
    search_urls = [
        search_url,
        "https://esgf-node.llnl.gov/esg-search",
        "https://esgf-index1.ceda.ac.uk/esg-search",
        "https://esgf.nci.org.au/esg-search"
    ]

    # List of alternative ESGF search nodes
    search_urls = [
        search_url,
        "https://esgf-node.llnl.gov/esg-search",
        "https://esgf-index1.ceda.ac.uk/esg-search",
        "https://esgf.nci.org.au/esg-search"
    ]

    # Try each ESGF server until we successfully download the dataset
    ds = None
    opendap_urls = []
    
    for server_idx, url in enumerate(search_urls):
        try:
            logger.info(f"🌐 Trying ESGF search node: {url} (attempt {server_idx + 1}/{len(search_urls)})")
            
            # Connect to ESGF and search for dataset
            conn = SearchConnection(url, distrib=distrib)
            
            # Use specific facets for CORDEX to avoid warning and improve search efficiency
            cordex_facets = 'project,domain,variable,experiment,time_frequency,model,downscaling_realisation'
            ctx = conn.new_context(facets=cordex_facets, query=f"id:{query}")
            hit_count = ctx.hit_count
            
            if hit_count == 0:
                logger.warning(f"No datasets found at {url}")
                continue
                
            logger.info(f"✅ Found {hit_count} datasets matching query at {url}")
            
            # Try to download from this server
            ds, opendap_urls = _download_from_server(ctx, split_by_variable, pydap, session)
            
            if ds is not None:
                logger.info(f"🎉 Successfully downloaded dataset from {url}")
                break  # Success!
            else:
                logger.warning(f"Found dataset but failed to download from {url}")
                
        except Exception as e:
            logger.warning(f"❌ Failed with server {url}: {type(e).__name__}: {e}")
            if server_idx == len(search_urls) - 1:  # Last server
                logger.error("❌ All ESGF servers failed")
                logger.error("Possible solutions:")
                logger.error("1. Check your internet connection")
                logger.error("2. Try again later - ESGF servers may be temporarily unavailable")
                logger.error("3. Verify the dataset ID is correct")
                raise ConnectionError("Unable to download dataset from any ESGF server") from e
            else:
                logger.info(f"🔄 Trying next ESGF server...")

    if ds is None:
        raise ConnectionError("Failed to download dataset from any ESGF server")

    return ds, opendap_urls


def _download_from_server(ctx, split_by_variable, pydap, session):
    """Helper function to download dataset from a specific ESGF server context."""
    try:
        results = ctx.search()
        logger.info(f"🔍 Processing {len(list(results))} result(s)...")
        results = ctx.search()  # Need to search again as generator is consumed

        if split_by_variable:
            ds = []

        opendap_urls = []
        multiple_opendap_urls = []

        # Process each result and extract OpenDAP URLs
        for result in results:
            if split_by_variable:
                current_opendap_urls = []

            nc_files = result.file_context().search()
            for nc_file in nc_files:
                if split_by_variable:
                    current_opendap_urls.append(nc_file.opendap_url)
                else:
                    opendap_urls.append(nc_file.opendap_url)

            # Open datasets based on protocol and splitting preference
            if split_by_variable:
                if pydap:
                    stores = build_pydap_stores(current_opendap_urls, session)
                    try:
                        ds.append(xr.open_mfdataset(stores, combine="by_coords"))
                    except Exception as e:
                        logger.error(f"❌ Failed to open dataset with PyDAP: {e}")
                        logger.info("🔄 Trying alternative approach without PyDAP...")
                        ds.append(xr.open_mfdataset(current_opendap_urls, combine="by_coords"))
                else:
                    try:
                        ds.append(xr.open_mfdataset(current_opendap_urls, combine="by_coords"))
                    except Exception as e:
                        logger.error(f"❌ Failed to open dataset from URLs: {e}")
                        raise

                multiple_opendap_urls.extend(current_opendap_urls)

        # Consolidate URLs if split by variable
        if split_by_variable:
            opendap_urls = multiple_opendap_urls
        else:
            if pydap:
                stores = build_pydap_stores(opendap_urls, session)
                try:
                    ds = xr.open_mfdataset(stores, combine="by_coords")
                except Exception as e:
                    logger.error(f"❌ Failed to open dataset with PyDAP: {e}")
                    logger.info("🔄 Trying alternative approach without PyDAP...")
                    ds = xr.open_mfdataset(opendap_urls, combine="by_coords")
            else:
                try:
                    ds = xr.open_mfdataset(opendap_urls, combine="by_coords")
                except Exception as e:
                    logger.error(f"❌ Failed to open dataset from URLs: {e}")
                    raise

        return ds, opendap_urls
        
    except Exception as e:
        logger.warning(f"Download failed from this server: {e}")
        return None, []


def build_pydap_stores(
    opendap_urls: list[str], session
) -> list[xr.backends.PydapDataStore]:
    """Build PyDAP data stores from OpenDAP URLs.

    Creates xarray-compatible PyDAP stores for each OpenDAP URL using
    an authenticated session.

    Args:
        opendap_urls (list[str]): List of OpenDAP URLs to access.
        session: Authenticated PyDAP session object.

    Returns:
        list[xr.backends.PydapDataStore]: List of PyDAP data stores ready
            for use with xarray.

    Example:
        >>> session = setup_session(openid, password)
        >>> urls = ["http://esgf.../file1.nc", "http://esgf.../file2.nc"]
        >>> stores = build_pydap_stores(urls, session)
        >>> ds = xr.open_mfdataset(stores, combine="by_coords")
    """
    stores = []
    for opendap_url in opendap_urls:
        # Remove .dods suffix if present
        store = xr.backends.PydapDataStore.open(
            opendap_url.rstrip(".dods"), session=session
        )
        stores.append(store)

    return stores


def search_and_download_cordex(
    query: dict,
    openid: str,
    password: str,
    output_folder: str,
    box: tuple = (),
    interpolate_grid: str | None = None,
    crop_suffix: str = "cropped",
    interpolate_suffix: str = "interpolated",
    remove_uncropped: bool = True,
    remove_uninterpolated: bool = True,
    hostname: str = "https://esg-dn1.nsc.liu.se/esg-search",
    distrib: bool = False,
    first_time: bool = True,
    facets: str | None = None,
) -> None:
    """Search and download CORDEX files with optional post-processing.

    Searches ESGF catalog, downloads matching files, and optionally crops
    to a bounding box and/or interpolates to a target grid.

    Args:
        query (dict): CORDEX query parameters (project, domain, variable, etc.).
        openid (str): ESGF OpenID URL for authentication.
        password (str): Password for ESGF authentication.
        output_folder (str): Directory where files will be saved.
        box (tuple, optional): Bounding box for cropping as
            (lon_min, lon_max, lat_min, lat_max). Defaults to empty tuple.
        interpolate_grid (str, optional): Target grid specification for CDO
            interpolation. Defaults to None.
        crop_suffix (str, optional): Suffix for cropped files. Defaults to "cropped".
        interpolate_suffix (str, optional): Suffix for interpolated files.
            Defaults to "interpolated".
        remove_uncropped (bool, optional): If True, removes original file after
            cropping. Defaults to True.
        remove_uninterpolated (bool, optional): If True, removes file after
            interpolation. Defaults to True.
        hostname (str, optional): ESGF search node URL. Defaults to
            "https://esg-dn1.nsc.liu.se/esg-search".
        distrib (bool, optional): If True, searches across distributed nodes.
            Defaults to False.
        first_time (bool, optional): If True, bootstraps certificates. Defaults to True.
        facets (str, optional): Comma-separated list of facets to include in search.
            If None, uses CORDEX-specific defaults to avoid warnings. Defaults to None.

    Example:
        >>> query = {"project": "CORDEX", "domain": "EUR-11", "variable": "tas"}
        >>> search_and_download_cordex(
        ...     query,
        ...     "https://esg-dn1.nsc.liu.se/esgf-idp/openid/user",
        ...     "password",
        ...     "./data",
        ...     box=(-10, 5, 35, 45)
        ... )
    """
    # Authenticate with ESGF
    lm = LogonManager()
    lm.logon_with_openid(openid=openid, password=password, bootstrap=first_time)

    # Search for matching datasets
    conn = SearchConnection(hostname, distrib=distrib)
    
    # Use specific facets if provided, otherwise use CORDEX defaults to avoid warning
    if facets is None:
        cordex_facets = 'project,domain,variable,experiment,time_frequency,model,downscaling_realisation'
        ctx = conn.new_context(facets=cordex_facets)
    else:
        ctx = conn.new_context(facets=facets)
    results = ctx.search(**query)

    # Download and process each file
    files = results[0].file_context().search()
    for f in files:
        filename = Path(f"{output_folder}/{f.filename}")
        
        if filename.is_file():
            logger.info(f"Skipping (already exists): {f.download_url}")
        else:
            download_file(f, output_folder)
            
            # Apply spatial cropping if requested
            if box:
                filename = crop_file(
                    filename, box, suffix=crop_suffix, remove_original=remove_uncropped
                )
            
            # Apply grid interpolation if requested
            if interpolate_grid:
                filename = interpolate_file(
                    filename,
                    interpolate_grid,
                    suffix=interpolate_suffix,
                    remove_original=remove_uninterpolated,
                )


def download_file(file_obj, output_path: str) -> None:
    """Download a single file from ESGF using HTTP with certificates.

    Downloads a file from ESGF servers using authentication certificates
    with progress bar display.

    Args:
        file_obj: ESGF file object containing download_url and filename attributes.
        output_path (str): Directory where the file will be saved.

    Raises:
        FileNotFoundError: If ESGF certificates are not found.
        SystemExit: If download fails or file size mismatch detected.

    Example:
        >>> # file_obj from ESGF search results
        >>> download_file(file_obj, "./data")
    """
    # Validate ESGF certificates exist
    _validate_esgf_dir()
    
    # Set up certificate paths
    cert = f"{ESGF_DIR}/credentials.pem"
    ca_certs = f"{ESGF_DIR}/certificates"

    headers = {"user-agent": "requests", "connection": "close"}

    # Initiate download with streaming
    response = requests.get(
        file_obj.download_url,
        cert=(cert, cert),
        verify=ca_certs,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=120,
    )

    if response.ok:
        logger.info(f"Downloading: {file_obj.download_url}")

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        wrote = 0

        # Download with progress bar
        with open(f"{output_path}/{file_obj.filename}", "wb") as output_file:
            for data in tqdm(
                response.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit="KB",
                unit_scale=True,
            ):
                wrote = wrote + len(data)
                output_file.write(data)
        
        # Verify download completed successfully
        if total_size != 0 and wrote != total_size:
            logger.error("Download incomplete: size mismatch")
            exit(-1)
    else:
        logger.error(f"Download failed: HTTP {response.status_code}")
        exit(-1)


def generate_output_name_with_suffix(file_path: Path, suffix: str) -> str:
    """Generate output filename by adding suffix before extension.

    Args:
        file_path (Path): Original file path.
        suffix (str): Suffix to add to filename.

    Returns:
        str: New filename with suffix added.

    Example:
        >>> from pathlib import Path
        >>> path = Path("/data/climate_data.nc")
        >>> generate_output_name_with_suffix(path, "cropped")
        '/data/climate_data_cropped.nc'
    """
    return f"{file_path.resolve().parent}/{file_path.stem}_{suffix}{file_path.suffix}"


def generate_output_name_with_prefix(file_path: Path, prefix: str) -> str:
    """Generate output filename by adding prefix before filename.

    Args:
        file_path (Path): Original file path.
        prefix (str): Prefix to add to filename.

    Returns:
        str: New filename with prefix added.

    Example:
        >>> from pathlib import Path
        >>> path = Path("/data/climate_data.nc")
        >>> generate_output_name_with_prefix(path, "processed")
        '/data/processed_climate_data.nc'
    """
    return f"{file_path.resolve().parent}/{prefix}_{file_path.stem}{file_path.suffix}"


def crop_file(
    file_path: Path,
    bounding_box: tuple[float, float, float, float],
    suffix: str = "cropped",
    prefix: str | None = None,
    remove_original: bool = True,
    cdo_path: str | None = None,
) -> Path:
    """Crop NetCDF file to geographic bounding box using CDO.

    Uses Climate Data Operators (CDO) to extract a geographic subset
    from a climate data file.

    Args:
        file_path (Path): Path to the input NetCDF file.
        bounding_box (tuple[float, float, float, float]): Geographic bounds as
            (lon_min, lon_max, lat_min, lat_max) in degrees.
        suffix (str, optional): Suffix for output filename. Defaults to "cropped".
        prefix (str, optional): Prefix for output filename. If provided, takes
            precedence over suffix. Defaults to None.
        remove_original (bool, optional): If True, deletes original file after
            cropping. Defaults to True.
        cdo_path (str, optional): Custom path to CDO executable. Defaults to None.

    Returns:
        Path: Path to the cropped output file.

    Example:
        >>> from pathlib import Path
        >>> input_file = Path("./data/temperature.nc")
        >>> # Crop to Iberian Peninsula
        >>> output = crop_file(input_file, (-10, 5, 35, 45))
        >>> print(output)
        ./data/temperature_cropped.nc
    """
    if cdo_path:
        cdo = Cdo(cdo_path)
    else:
        cdo = Cdo()

    input_name = str(file_path.resolve())
    
    # Generate output filename
    if prefix is not None:
        output_name = str(generate_output_name_with_prefix(file_path, prefix))
    else:
        output_name = str(generate_output_name_with_suffix(file_path, suffix))

    # Apply spatial cropping
    cdo.sellonlatbox(
        bounding_box[0],
        bounding_box[1],
        bounding_box[2],
        bounding_box[3],
        input=input_name,
        output=output_name,
        options="-z zip",
    )

    # Remove original file if requested
    if remove_original:
        file_path.resolve().unlink()

    # Return path to output file
    if prefix is not None:
        output_path = Path(generate_output_name_with_prefix(file_path, prefix))
    else:
        output_path = Path(generate_output_name_with_suffix(file_path, suffix))

    return output_path


def interpolate_file(
    file_path: Path,
    target_grid: str,
    suffix: str = "interpolated",
    remove_original: bool = True,
    cdo_path: str | None = None,
) -> Path:
    """Interpolate NetCDF file to target grid using CDO.

    Uses Climate Data Operators (CDO) to regrid climate data to a
    specified target grid using distance-weighted interpolation.

    Args:
        file_path (Path): Path to the input NetCDF file.
        target_grid (str): Target grid specification for CDO (grid description
            file path or grid specification string).
        suffix (str, optional): Suffix for output filename. Defaults to "interpolated".
        remove_original (bool, optional): If True, deletes original file after
            interpolation. Defaults to True.
        cdo_path (str, optional): Custom path to CDO executable. Defaults to None.

    Returns:
        Path: Path to the interpolated output file.

    Example:
        >>> from pathlib import Path
        >>> input_file = Path("./data/temperature.nc")
        >>> # Interpolate to 0.25-degree grid
        >>> output = interpolate_file(input_file, "r1440x720")
        >>> print(output)
        ./data/temperature_interpolated.nc
    """
    if cdo_path:
        cdo = Cdo(cdo_path)
    else:
        cdo = Cdo()

    input_name = str(file_path.resolve())
    output_name = str(generate_output_name_with_suffix(file_path, suffix))

    # Apply grid interpolation using distance-weighted average
    cdo.remapdis(target_grid, input=input_name, output=output_name, options="-z zip")

    # Remove original file if requested
    if remove_original:
        file_path.resolve().unlink()

    return Path(generate_output_name_with_suffix(file_path, suffix))