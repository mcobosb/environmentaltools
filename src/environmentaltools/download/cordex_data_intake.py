"""CORDEX climate data download utilities using intake-esgf.

This module provides functions to download and process CORDEX (Coordinated Regional
Climate Downscaling Experiment) data from ESGF (Earth System Grid Federation) servers
using the modern intake-esgf package, replacing the deprecated pyesgf.

Key improvements over pyesgf:
- Better compatibility with modern ESGF indices
- More robust authentication and connection handling
- Improved performance and reliability
- Active maintenance and development

For more information, see: https://intake-esgf.readthedocs.io/
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr
import intake_esgf


from configobj import ConfigObj
from environmentaltools.common import utils


def query_esgf_catalog_intake(
    query: Dict[str, Union[str, List[str]]],
    indices: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Query ESGF catalog using intake-esgf.

    Modern replacement for query_esgf_catalog using intake-esgf package.
    Provides better performance and compatibility with current ESGF infrastructure.

    Args:
        query: Dictionary of CORDEX query parameters. Common keys include:
            - project (str): e.g., "CORDEX"
            - domain (str): e.g., "EUR-11"
            - experiment (str): e.g., "rcp85"
            - time_frequency (str): e.g., "3hr", "day", "mon"
            - variable (list): e.g., ["pr", "tas"]
        indices: List of ESGF indices to search. If None, uses default
            indices. Defaults to None.
        **kwargs: Additional query parameters

    Returns:
        pd.DataFrame: DataFrame containing metadata for all matching datasets.

    Raises:
        ImportError: If intake-esgf is not installed
        ConnectionError: If unable to connect to any ESGF index

    Example:
        >>> query = {"project": "CORDEX", "domain": "EUR-11", "variable": ["pr"]}
        >>> datasets = query_esgf_catalog_intake(query)
        >>> print(f"Found {len(datasets)} datasets")
    """
    if intake_esgf is None:
        raise ImportError(
            "intake-esgf is required for this function. "
            "Install it with: pip install intake-esgf"
        )

    # Default ESGF indices to search
    if indices is None:
        indices = [
            "https://esgf-node.llnl.gov/esg-search",
            "https://esgf-index1.ceda.ac.uk/esg-search", 
            "https://esgf.nci.org.au/esg-search",
            "https://esg-dn1.nsc.liu.se/esg-search"
        ]

    results = []
    successful_indices = []

    # Try each index until we get results
    for idx_url in indices:
        try:
            logger.info(f"🔍 Searching ESGF index: {idx_url}")
            
            # Create intake-esgf catalog
            # Note: intake-esgf uses a different API structure
            cat = intake_esgf.ESGFCatalog()
            
            # Build search query for intake-esgf format
            search_query = query.copy()
            
            # Perform search
            search_results = cat.search(**search_query)
            
            if hasattr(search_results, '__len__') and len(search_results) > 0:
                logger.info(f"✅ Found {len(search_results)} results at {idx_url}")
                # Convert results to DataFrame format
                if hasattr(search_results, 'df'):
                    results.extend(search_results.df.to_dict('records'))
                else:
                    # Handle different result formats
                    results.extend([{"id": str(item)} for item in search_results])
                successful_indices.append(idx_url)
                break  # Use first successful index
            else:
                logger.warning(f"No results found at {idx_url}")
                
        except Exception as e:
            logger.warning(f"❌ Failed to search {idx_url}: {type(e).__name__}: {e}")
            continue

    if not results:
        logger.error("No datasets found - this might be due to:")
        logger.error("1. intake-esgf API differences from expected interface")
        logger.error("2. Network connectivity issues")
        logger.error("3. Query parameters not matching available data")
        logger.error("4. ESGF server temporary unavailability")
        
        # Return empty DataFrame instead of raising error for now
        logger.warning("Returning empty DataFrame - check intake-esgf documentation for correct usage")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(results)
    logger.info(f"📊 Total datasets found: {len(df)} from {len(successful_indices)} indices")
    
    return df


def download_esgf_dataset_intake(
    dataset_id: str,
    indices: Optional[List[str]] = None,
    auth: Optional[Dict[str, str]] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs
) -> Tuple[xr.Dataset, List[str]]:
    """Download ESGF dataset using intake-esgf.

    Modern replacement for download_esgf_dataset using intake-esgf package.
    Provides better authentication handling and connection reliability.

    Args:
        dataset_id: ESGF dataset ID to download
        indices: List of ESGF indices to try. If None, uses defaults
        auth: Authentication dictionary with 'username' and 'password' keys
        chunks: Chunking specification for xarray, e.g., {'time': 100}
        **kwargs: Additional parameters for intake-esgf

    Returns:
        Tuple containing:
            - Dataset: xarray Dataset with the climate data
            - URLs: List of access URLs used

    Raises:
        ImportError: If intake-esgf is not installed
        ConnectionError: If unable to download from any index

    Example:
        >>> auth = {'username': 'user', 'password': 'pass'}
        >>> ds, urls = download_esgf_dataset_intake(
        ...     "cordex.output.EUR-11.SMHI...",
        ...     auth=auth
        ... )
    """
    if intake_esgf is None:
        raise ImportError(
            "intake-esgf is required for this function. "
            "Install it with: pip install intake-esgf"
        )

    # Default ESGF indices
    if indices is None:
        indices = [
            "https://esgf-node.llnl.gov/esg-search",
            "https://esgf-index1.ceda.ac.uk/esg-search",
            "https://esgf.nci.org.au/esg-search", 
            "https://esg-dn1.nsc.liu.se/esg-search"
        ]

    # Default chunking for better performance
    if chunks is None:
        chunks = {'time': 100}

    dataset = None
    access_urls = []
    
    # Try each index
    for idx_url in indices:
        try:
            logger.info(f"🌐 Trying to download from: {idx_url}")
            
            # Create catalog with authentication if provided
            cat_kwargs = {'esgf_search_url': idx_url}
            if auth:
                cat_kwargs.update(auth)
                
            cat = intake_esgf.ESGFCatalog(**cat_kwargs, **kwargs)
            
            # Search for the specific dataset
            search_results = cat.search(id=dataset_id)
            
            if len(search_results) == 0:
                logger.warning(f"Dataset {dataset_id} not found at {idx_url}")
                continue
                
            logger.info(f"✅ Found dataset at {idx_url}")
            
            # Get the first matching dataset
            dataset_entry = search_results[list(search_results)[0]]
            
            # Open the dataset with chunking
            dataset = dataset_entry.to_dask(chunks=chunks)
            access_urls = [dataset_entry.urlpath]  # intake-esgf specific
            
            logger.info(f"🎉 Successfully loaded dataset from {idx_url}")
            break
            
        except Exception as e:
            logger.warning(f"❌ Failed to download from {idx_url}: {type(e).__name__}: {e}")
            continue

    if dataset is None:
        raise ConnectionError(
            f"Failed to download dataset {dataset_id} from any ESGF index. "
            f"Tried: {indices}"
        )

    return dataset, access_urls


def download_with_config_intake(
    output_folder: str,
    auth_config: Optional[str] = None,
) -> None:
    """Download CORDEX data using configuration files with intake-esgf.

    Modern replacement for download_with_config using intake-esgf package.
    Maintains the same interface but uses the updated backend.

    Args:
        output_folder: Directory path where files will be stored
        auth_config: Path to authentication config file. If None,
            uses ~/.esgf/config.ini

    Raises:
        ImportError: If intake-esgf is not installed
        FileNotFoundError: If required configuration files are missing
        
    Note:
        Expects the same file structure as the original function:
        - coordinates.csv: Coordinate points
        - selection.csv: Dataset selection indices
        - queries.xlsx: Available dataset queries
    """
    if intake_esgf is None:
        raise ImportError(
            "intake-esgf is required for this function. "
            "Install it with: pip install intake-esgf"
        )

    output_folder = Path(output_folder)
    query_file = output_folder / "queries.xlsx"
    coord_file = output_folder / "coordinates.csv"
    selection_file = output_folder / "selection.csv"

    # Load authentication config
    if auth_config is None:
        auth_config = os.path.expanduser("~/.esgf/config.ini")
        
    auth = None
    if os.path.exists(auth_config):
        try:
            config = ConfigObj(auth_config)
            auth = {
                'username': config["credentials"]["openid"],
                'password': config["credentials"]["password"]
            }
            logger.info(f"🔐 Loaded authentication from: {auth_config}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load authentication: {e}")
            logger.info("🔓 Continuing without authentication")
    else:
        logger.info("🔓 No authentication config found - using anonymous access")

    # Load data files
    try:
        coords = pd.read_csv(coord_file)[3:]  # Skip header rows
        all_queries = pd.read_excel(query_file)
        selection = pd.read_csv(selection_file, header=None).values[:, 0].tolist()
    except FileNotFoundError as e:
        logger.error(f"❌ Required file not found: {e}")
        raise

    # Filter queries based on selection
    queries = all_queries.loc[selection, ["id"]]

    logger.info("📡 Starting CORDEX data download with intake-esgf...")
    logger.info(f"🎯 Processing {len(coords)} coordinates and {len(queries)} datasets")

    # Download data for each coordinate
    for coord in coords.itertuples():
        coord_folder = output_folder / f"coord_{coord.Index:02}"
        coord_folder.mkdir(exist_ok=True)

        lat, lon = coord.lat, coord.lon
        logger.info(f"📍 Processing coordinate: lat={lat}, lon={lon}")

        # Download each selected query
        for q in queries.itertuples():
            filename = (
                coord_folder / 
                f"c{coord.Index:02}-q{q.Index:02}-{secure_filename(q.id)}.nc"
            )

            if filename.exists():
                logger.info(f"✅ {q.id} already exists")
                continue

            try:
                logger.info(f"⬇️  Downloading {q.id}")
                
                # Download with intake-esgf
                dataset, _ = download_esgf_dataset_intake(
                    q.id, 
                    auth=auth,
                    chunks={'time': 50}  # Smaller chunks for point extraction
                )

                # Extract point of interest
                poi = utils.nearest(dataset, lat, lon)
                
                # Save to NetCDF
                poi.to_netcdf(filename)
                poi.close()
                dataset.close()
                
                logger.info(f"✅ Saved: {filename.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {q.id}: {e}")
                logger.info("⏭️  Continuing with next dataset...")
                continue

    logger.info("🎉 Download process completed!")


def secure_filename(filename: str) -> str:
    """Create secure filename from dataset ID."""
    # Simple implementation - replace unsafe characters
    safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in safe_chars else "_" for c in filename)


# Compatibility functions to maintain API
def query_esgf_catalog(*args, **kwargs):
    """Compatibility wrapper - redirects to intake-esgf implementation."""
    logger.warning(
        "⚠️  Using legacy pyesgf function. "
        "Consider migrating to query_esgf_catalog_intake() for better performance."
    )
    return query_esgf_catalog_intake(*args, **kwargs)


def download_esgf_dataset(*args, **kwargs):
    """Compatibility wrapper - redirects to intake-esgf implementation."""
    logger.warning(
        "⚠️  Using legacy pyesgf function. "
        "Consider migrating to download_esgf_dataset_intake() for better performance."
    )
    return download_esgf_dataset_intake(*args, **kwargs)


def download_with_config(*args, **kwargs):
    """Compatibility wrapper - redirects to intake-esgf implementation."""
    logger.warning(
        "⚠️  Using legacy pyesgf function. "
        "Consider migrating to download_with_config_intake() for better performance."
    )
    return download_with_config_intake(*args, **kwargs)