"""CORDEX climate data download utilities.

This module provides functions to download and process CORDEX (Coordinated Regional
Climate Downscaling Experiment) data from ESGF (Earth System Grid Federation) servers.
"""

import asyncio
import logging
import math
import os
import sys
from pathlib import Path

# Set HOME environment variable for cross-platform compatibility
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser("~")

import numpy as np
import pandas as pd
import requests
import xarray as xr

# Optional dependencies for CORDEX download
try:
    from cdo import Cdo
    HAS_CDO = True
except ImportError:
    HAS_CDO = False
    Cdo = None

try:
    from configobj import ConfigObj
    HAS_CONFIGOBJ = True
except ImportError:
    HAS_CONFIGOBJ = False
    ConfigObj = None

try:
    from pydap.cas.esgf import setup_session
    from pyesgf.logon import LogonManager
    from pyesgf.search import SearchConnection
    HAS_ESGF = True
except ImportError:
    HAS_ESGF = False
    setup_session = None
    LogonManager = None
    SearchConnection = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple pass-through if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from werkzeug.utils import secure_filename
    HAS_WERKZEUG = True
except ImportError:
    HAS_WERKZEUG = False
    # Fallback: simple filename sanitization
    def secure_filename(filename):
        return "".join(c for c in filename if c.isalnum() or c in "._- ")

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


# Set up module logger
logger = logging.getLogger(__name__)


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


# Additional imports for advanced functionality



def batch_download_with_config(
    output_folder: str,
    download: bool = True,
    bootstrap: bool = False,
    pydap: bool = False,
) -> None:
    """Batch download CORDEX data based on configuration files.

    This function provides a workflow for querying and downloading CORDEX data
    using configuration files that specify coordinates and query selections.

    Args:
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
        - Expects a config.ini file with ESGF credentials
        - Requires CSV files: {output_folder}_coordenadas.csv (coordinates)
          and {output_folder}_seleccion.csv (query selection)
        - Creates output structure: output_folder/coord_XX/files.nc

    Example:
        >>> # Generate queries only
        >>> batch_download_with_config('./data', download=False)
        >>> # Download data after selecting queries
        >>> batch_download_with_config('./data', download=True)
    """
    # Define default query parameters for CORDEX EUR-11 domain
    query = {
        "project": "CORDEX",
        "domain": "EUR-11",
        "experiment": "rcp85",
        "time_frequency": "3hr",
        "variable": ["pr"],
    }

    # Create output folder structure
    Path(output_folder).mkdir(exist_ok=True)
    query_file = output_folder + "_queries.xlsx"

    if download:
        # Load configuration and data files
        coord_file = output_folder + "_coordenadas.csv"
        selection_file = output_folder + "_seleccion.csv"

        # Read ESGF credentials from config file
        config = ConfigObj("config.ini")
        openid = config["credentials"]["openid"]
        password = config["credentials"]["password"]

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
    else:
        # Generate query file without downloading
        logger.info("Generating query file")
        all_queries = query_esgf_catalog(query)
        all_queries.to_excel(f"{output_folder}/{query_file}")

    logger.info("Finished!")


def query_esgf_catalog(
    query: dict,
    search_url: str = "https://esg-dn1.nsc.liu.se/esg-search",
    distrib: bool = True,
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

    Returns:
        pd.DataFrame: DataFrame containing metadata for all matching datasets.
            Each row represents one dataset with its full metadata.

    Example:
        >>> query = {"project": "CORDEX", "domain": "EUR-11", "variable": ["pr"]}
        >>> datasets = query_esgf_catalog(query)
        >>> print(f"Found {len(datasets)} datasets")
    """
    queries = {}

    # Connect to ESGF search API
    conn = SearchConnection(search_url, distrib=distrib)
    ctx = conn.new_context(**query)
    
    # Retrieve all matching results
    if ctx.hit_count > 0:
        results = ctx.search()
        for i, result in enumerate(results):
            queries[i] = result.json

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame.from_dict(queries, orient="index")
    return df


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

    Args:
        query (str): ESGF dataset ID to download.
        search_url (str, optional): ESGF search node URL. Defaults to
            "https://esg-dn1.nsc.liu.se/esg-search".
        distrib (bool, optional): If True, searches across distributed nodes.
            Defaults to True.
        split_by_variable (bool, optional): If True, returns separate datasets
            for each variable. Defaults to False.
        openid (str, optional): ESGF OpenID URL for authentication. Defaults to None.
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
        if pydap:
            session = setup_session(openid, password, check_url=openid)
        else:
            lm = LogonManager()
            lm.logon_with_openid(openid=openid, password=password, bootstrap=bootstrap)

    # Connect to ESGF and search for dataset
    conn = SearchConnection(search_url, distrib=distrib)

    ctx = conn.new_context(query=f"id:{query}")
    hit_count = ctx.hit_count
    
    if hit_count > 0:
        logger.info(f"Found {hit_count} matching files")
        results = ctx.search()

        if split_by_variable:
            ds = []

        # Process each result and extract OpenDAP URLs
        for result in results:
            if split_by_variable:
                opendap_urls = []

            nc_files = result.file_context().search()
            for nc_file in nc_files:
                opendap_urls.append(nc_file.opendap_url)

            # Open datasets based on protocol and splitting preference
            if split_by_variable:
                if pydap:
                    stores = build_pydap_stores(opendap_urls, session)
                    ds.append(xr.open_mfdataset(stores, combine="by_coords"))
                else:
                    ds.append(xr.open_mfdataset(opendap_urls, combine="by_coords"))

                multiple_opendap_urls.extend(opendap_urls)

        # Consolidate URLs if split by variable
        if split_by_variable:
            opendap_urls = multiple_opendap_urls
        else:
            if pydap:
                stores = build_pydap_stores(opendap_urls, session)
                ds = xr.open_mfdataset(stores, combine="by_coords")
            else:
                ds = xr.open_mfdataset(opendap_urls, combine="by_coords")

    return ds, opendap_urls


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
    ctx = conn.new_context()
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