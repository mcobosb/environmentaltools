"""ERA5 data download and processing system.

This module provides functions to download and process data from the ERA5
reanalysis dataset using the Climate Data Store (CDS) API. Includes functionality
for automated data retrieval, preprocessing, and export to various formats.

The module supports downloading any ERA5 variable (marine, atmospheric, etc.)
with configurable spatial and temporal domains.

Features:
    - Automated ERA5 data download via CDS API
    - Support for any ERA5 variable (waves, wind, temperature, etc.)
    - Batch processing for multiple years
    - Configurable spatial and temporal domains
    - NetCDF to CSV conversion with pandas integration
    - Robust error handling and logging
    - Progress tracking for long downloads

Dependencies:
    - cdsapi: Climate Data Store API client
    - pandas: Data manipulation and analysis
    - environmentaltools: Data utilities

Note:
    Requires valid CDS API credentials configured in ~/.cdsapirc
    or environment variables CDSAPI_URL and CDSAPI_KEY

Example:
    >>> from environmentaltools.download import marine_copernicus
    >>> 
    >>> # Download significant wave height
    >>> config = {
    ...     'start_year': 2015,
    ...     'end_year': 2020,
    ...     'area_bounds': [41.4, -9.0, 41.0, -8.65],
    ...     'output_directory': './era5_data',
    ...     'variable': 'significant_height_of_combined_wind_waves_and_swell'
    ... }
    >>> results = marine_copernicus.download_era5_data(config)
    >>> 
    >>> # Or download 10m wind speed
    >>> config['variable'] = '10m_u_component_of_wind'
    >>> results = marine_copernicus.download_era5_data(config)
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cdsapi
import pandas as pd
from loguru import logger

from environmentaltools.common import read


class ERA5DataDownloadConfig:
    """Configuration class for ERA5 data download parameters.
    
    This class accepts configuration as a dictionary and provides validation
    and default values for all parameters. It supports any ERA5 variable
    (marine, atmospheric, land, etc.).
    
    Attributes:
        dataset_name (str): CDS dataset identifier.
        variable (str): Variable to download (e.g., wave height, wind, temperature).
        start_year (int): First year to download.
        end_year (int): Last year to download.
        area_bounds (list): Geographic bounds [North, West, South, East].
        output_directory (str): Directory for output files.
        file_prefix (str): Prefix for output filenames.
        retry_attempts (int): Number of retry attempts for failed downloads.
    
    Example:
        >>> # Download wave data
        >>> config_dict = {
        ...     'start_year': 2015,
        ...     'end_year': 2020,
        ...     'area_bounds': [41.4, -9.0, 41.0, -8.65],
        ...     'output_directory': './era5_data',
        ...     'variable': 'significant_height_of_combined_wind_waves_and_swell'
        ... }
        >>> config = ERA5DataDownloadConfig(config_dict)
        >>> 
        >>> # Download wind data
        >>> config_dict['variable'] = '10m_u_component_of_wind'
        >>> config = ERA5DataDownloadConfig(config_dict)
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize configuration from dictionary.
        
        Args:
            config (dict, optional): Configuration dictionary. If None, uses defaults.
                Supported keys:
                - dataset_name (str): Dataset identifier
                - variable (str): Variable name
                - start_year (int): Start year
                - end_year (int): End year
                - months (list): Months to download (default: all)
                - days (list): Days to download (default: all)
                - hours (list): Hours to download (default: all)
                - area_bounds (list): [North, West, South, East]
                - output_directory (str): Output path
                - retry_attempts (int): Number of retry attempts (default: 3)
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if config is None:
            config = {}
        
        self._parse_config(config)
        self._validate_config()
        logger.info("Configuration initialized successfully")
    
    def _parse_config(self, config: dict[str, Any]) -> None:
        """Parse configuration dictionary and set attributes with defaults."""
        # Dataset configuration
        self.dataset_name: str = config.get(
            'dataset_name', 'reanalysis-era5-single-levels'
        )
        self.product_type: str = config.get('product_type', 'reanalysis')
        self.format: str = config.get('format', 'netcdf')
        self.variable: str = config.get(
            'variable', 'significant_height_of_combined_wind_waves_and_swell'
        )
        
        # Temporal configuration
        self.start_year: int = config.get('start_year', 1981)
        self.end_year: int = config.get('end_year', 2018)
        self.months: list[str] = config.get(
            'months', [f'{i:02d}' for i in range(1, 13)]
        )
        self.days: list[str] = config.get(
            'days', [f'{i:02d}' for i in range(1, 32)]
        )
        self.hours: list[str] = config.get(
            'hours', [f'{i:02d}:00' for i in range(24)]
        )
        
        # Spatial configuration
        self.area_bounds: list[float] = config.get(
            'area_bounds', [41.4, -9.0, 41.0, -8.65]
        )
        
        # Output configuration
        self.output_directory: str = config.get('output_directory', "./data")
        self.file_prefix: str = config.get('file_prefix', "waves")
        
        # Download configuration
        self.retry_attempts: int = config.get('retry_attempts', 3)
        
    
    def _validate_config(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If critical configuration parameters are invalid.
        """
        # Validate temporal parameters
        if self.start_year > self.end_year:
            raise ValueError(
                f"Start year ({self.start_year}) cannot be greater than "
                f"end year ({self.end_year})"
            )
        
        if self.start_year < 1979:
            logger.warning(
                f"Start year {self.start_year} is before ERA5 availability (1979)"
            )
        
        current_year = datetime.now().year
        if self.end_year > current_year:
            logger.warning(
                f"End year {self.end_year} is in the future, "
                f"adjusting to {current_year}"
            )
            self.end_year = current_year
        
        # Validate spatial parameters
        if len(self.area_bounds) != 4:
            raise ValueError(
                "Area bounds must contain exactly 4 values [North, West, South, East]"
            )
        
        north, west, south, east = self.area_bounds
        if not (-90 <= south <= north <= 90):
            raise ValueError(
                f"Invalid latitude bounds: South={south}, North={north}"
            )
        
        if not (-180 <= west <= 360 and -180 <= east <= 360):
            raise ValueError(
                f"Invalid longitude bounds: West={west}, East={east}"
            )
        
        # Create output directory
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Configuration validation completed successfully")
    
    def print_summary(self) -> None:
        """Print a summary of the current configuration using loguru logger."""
        logger.info("\n" + "="*60)
        logger.info("ERA5 Data Download Configuration Summary")
        logger.info("="*60)
        logger.info(f"Dataset:         {self.dataset_name}")
        logger.info(f"Variable:        {self.variable}")
        logger.info(f"Product Type:    {self.product_type}")
        logger.info(f"Format:          {self.format}")
        logger.info(f"Time Period:     {self.start_year} - {self.end_year}")
        logger.info(f"Months:          {len(self.months)} months ({self.months[0]} to {self.months[-1]})")
        logger.info(f"Days:            {len(self.days)} days ({self.days[0]} to {self.days[-1]})")
        logger.info(f"Hours:           {len(self.hours)} hours ({self.hours[0]} to {self.hours[-1]})")
        logger.info(f"Area bounds:     [N:{self.area_bounds[0]}, W:{self.area_bounds[1]}, S:{self.area_bounds[2]}, E:{self.area_bounds[3]}]")
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"File prefix:     {self.file_prefix}")
        logger.info(f"Retry attempts:  {self.retry_attempts}")
        logger.info("="*60)
    

class ERA5DataDownloader:
    """Main class for downloading ERA5 data from CDS.
    
    This class handles the complete download workflow from any ERA5 dataset
    (marine, atmospheric, land variables, etc.) with robust error handling
    and progress tracking.
    
    Attributes:
        config (ERA5DataDownloadConfig): Configuration object.
        client (cdsapi.Client): CDS API client for downloads.
    
    Example:
        >>> config = ERA5DataDownloadConfig({'start_year': 2018, ...})
        >>> downloader = ERA5DataDownloader(config)
        >>> results = downloader.download_all_years()
        >>> print(f"Downloaded {sum(results.values())} years")
    """
    
    def __init__(self, config: ERA5DataDownloadConfig):
        """Initialize the downloader with configuration.
        
        Args:
            config (ERA5DataDownloadConfig): Configuration object.
        
        Raises:
            ConnectionError: If CDS API client cannot be initialized.
        """
        self.config = config
        self.client = None
        self._initialize_client()
        self._setup_output_directory()
    
    def _initialize_client(self) -> None:
        """
        Initialize the CDS API client with error handling.
        
        Raises
        ------
        ConnectionError
            If CDS API client cannot be initialized
        """
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {str(e)}")
            raise ConnectionError(f"CDS API initialization failed: {str(e)}")
    
    def _setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        output_path = Path(self.config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory configured: {output_path.absolute()}")
    
    def generate_year_list(self) -> list[str]:
        """
        Generate list of years to download based on configuration.
        
        Returns
        -------
        list[str]
            List of years as strings
        """
        years = [str(year) for year in range(self.config.start_year, self.config.end_year + 1)]
        logger.info(f"Generated year list: {self.config.start_year}-{self.config.end_year} ({len(years)} years)")
        return years
    
    def create_download_request(self, year: str) -> dict[str, Any]:
        """
        Create CDS API request dictionary for a specific year.
        
        Parameters
        ----------
        year : str
            Year to download data for
            
        Returns
        -------
        Dict[str, Any]
            CDS API request dictionary
        """
        request = {
            'product_type': self.config.product_type,
            'format': self.config.format,
            'variable': self.config.variable,
            'year': year,
            'month': self.config.months,
            'day': self.config.days,
            'time': self.config.hours,
            'area': self.config.area_bounds
        }
        
        logger.debug(f"Created download request for year {year}")
        return request
    
    def generate_output_filename(self, year: str) -> str:
        """
        Generate output filename for a specific year.
        
        Parameters
        ----------
        year : str
            Year for the filename
            
        Returns
        -------
        str
            Complete file path for output
        """
        filename = f"{self.config.file_prefix}_{year}.nc"
        filepath = Path(self.config.output_directory) / filename
        return str(filepath)
    
    def download_year_data(self, year: str, attempt: int = 1) -> bool:
        """
        Download wave data for a specific year with retry logic.
        Skip download if file already exists and is valid.
        
        Parameters
        ----------
        year : str
            Year to download data for
        attempt : int, optional
            Current attempt number (default: 1)
            
        Returns
        -------
        bool
            True if download successful or file already exists, False otherwise
        """
        try:
            # Create request and output filename
            request = self.create_download_request(year)
            output_file = self.generate_output_filename(year)
            
            # Enhanced file existence check
            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
                
                # Check if file size is reasonable (not empty or too small)
                if file_size >= self.config.min_file_size_mb:
                    logger.info(f"File already exists for year {year} ({file_size:.2f} MB) - skipping download")
                    return True
                else:
                    logger.warning(f"Existing file for year {year} is too small ({file_size:.2f} MB) - re-downloading")
                    # Remove the invalid file
                    try:
                        Path(output_file).unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove invalid file: {e}")
            
            logger.info(f"Starting download for year {year} (attempt {attempt}/{self.config.retry_attempts})")
            
            # Start download
            start_time = time.time()
            self.client.retrieve(self.config.dataset_name, request, output_file)
            
            # Verify download
            if Path(output_file).exists() and Path(output_file).stat().st_size > 0:
                elapsed_time = time.time() - start_time
                file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
                
                # Additional validation
                if file_size >= self.config.min_file_size_mb:
                    logger.info(f"Successfully downloaded {year} ({file_size:.2f} MB in {elapsed_time:.1f}s)")
                    return True
                else:
                    logger.error(f"Downloaded file for year {year} is too small ({file_size:.2f} MB)")
                    # Remove the invalid file
                    try:
                        Path(output_file).unlink()
                    except Exception:
                        pass
                    return False
            else:
                logger.error(f"Download verification failed for year {year}")
                return False
                
        except Exception as e:
            logger.error(f"Download failed for year {year} (attempt {attempt}): {str(e)}")
            
            # Retry logic
            if attempt < self.config.retry_attempts:
                logger.info(f"Retrying download for year {year} in {self.config.retry_delay} seconds...")
                time.sleep(self.config.retry_delay)
                return self.download_year_data(year, attempt + 1)
            else:
                logger.error(f"All retry attempts exhausted for year {year}")
                return False
    
    def download_all_years(self) -> dict[str, bool]:
        """
        Download wave data for all configured years.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping years to download success status
        """
        years = self.generate_year_list()
        download_results = {}
        
        logger.info(f"Starting batch download for {len(years)} years")
        start_time = time.time()
        
        for i, year in enumerate(years, 1):
            logger.info(f"Processing year {year} ({i}/{len(years)})")
            success = self.download_year_data(year)
            download_results[year] = success
            
            # Progress update
            if i % 5 == 0 or i == len(years):
                successful_downloads = sum(download_results.values())
                logger.info(f"Progress: {i}/{len(years)} years processed, "
                           f"{successful_downloads} successful downloads")
        
        # Summary
        total_time = time.time() - start_time
        successful_count = sum(download_results.values())
        failed_years = [year for year, success in download_results.items() if not success]
        
        logger.info(f"Download batch completed in {total_time:.1f}s")
        logger.info(f"Successful downloads: {successful_count}/{len(years)}")
        
        if failed_years:
            logger.warning(f"Failed downloads for years: {failed_years}")
        
        return download_results
    
    def validate_downloaded_files(self) -> dict[str, dict[str, Any]]:
        """
        Validate all downloaded NetCDF files.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary with validation results for each file
        """
        logger.info("Validating downloaded files...")
        validation_results = {}
        
        pattern = f"{self.config.file_prefix}_*.nc"
        nc_files = list(Path(self.config.output_directory).glob(pattern))
        
        for nc_file in nc_files:
            try:
                # Extract year from filename
                year = nc_file.stem.split('_')[-1]
                
                # Basic file validation
                file_stats = nc_file.stat()
                file_size_mb = file_stats.st_size / (1024 * 1024)
                
                validation_results[year] = {
                    'file_exists': True,
                    'file_size_mb': file_size_mb,
                    'file_path': str(nc_file),
                    'last_modified': datetime.fromtimestamp(file_stats.st_mtime),
                    'valid': file_size_mb > 0.1  # Basic size check
                }
                
                logger.debug(f"Year {year}: {file_size_mb:.2f} MB - Valid")
                
            except Exception as e:
                logger.error(f"Validation failed for {nc_file}: {str(e)}")
                validation_results[year] = {
                    'file_exists': False,
                    'error': str(e),
                    'valid': False
                }
        
        valid_files = sum(1 for result in validation_results.values() if result['valid'])
        logger.info(f"File validation completed: {valid_files}/{len(validation_results)} files valid")
        
        return validation_results


class ERA5DataProcessor:
    """Class for processing downloaded ERA5 data files.
    
    Handles conversion from NetCDF to various formats and data aggregation
    for any ERA5 variable (marine, atmospheric, etc.).
    
    Attributes:
        config (ERA5DataDownloadConfig): Configuration object.
    
    Example:
        >>> processor = ERA5DataProcessor(config)
        >>> data = processor.load_netcdf_files()
        >>> csv_file = processor.export_to_csv(data)
    """
    
    def __init__(self, config: ERA5DataDownloadConfig):
        """Initialize the processor with configuration.
        
        Args:
            config (ERA5DataDownloadConfig): Configuration object.
        """
        self.config = config
    
    def load_netcdf_files(self, file_pattern: Optional[str] = None) -> pd.DataFrame:
        """
        Load and concatenate multiple NetCDF files into a pandas DataFrame.
        
        Parameters
        ----------
        file_pattern : str, optional
            Glob pattern for NetCDF files (default: "\\*_waves.nc")
            
        Returns
        -------
        pd.DataFrame
            Concatenated wave data with time index
            
        Raises
        ------
        FileNotFoundError
            If no NetCDF files are found
        ValueError
            If data loading or processing fails
        """
        try:
            if file_pattern is None:
                file_pattern = f"{self.config.file_prefix}_*.nc"
            
            logger.info(f"Loading NetCDF files with pattern: {file_pattern}")
            
            # Change to output directory for file operations
            original_cwd = os.getcwd()
            os.chdir(self.config.output_directory)
            
            try:
                # Load data using environmentaltools
                data = read.netcdf(
                    file_pattern, 
                    ["swh"], 
                    glob=True, 
                    time_series=False
                )
                
                # Convert to pandas DataFrame
                if hasattr(data, 'swh') and (hasattr(data, 'time') or hasattr(data, 'valid_time')):
                    # Extract significant wave height data
                    swh_values = data.swh.values
                    
                    # Handle multi-dimensional data (take first spatial point if needed)
                    if swh_values.ndim > 1:
                        swh_values = swh_values[:, 0, 0]  # Take first lat, lon point
                    
                    # Get time coordinate (try both 'time' and 'valid_time')
                    if hasattr(data, 'valid_time'):
                        time_values = data.valid_time.values
                    else:
                        time_values = data.time.values
                    
                    # Create DataFrame with time index
                    df = pd.DataFrame(
                        swh_values, 
                        index=time_values, 
                        columns=["swh"]
                    )
                    
                    # Clean data: remove NaN values
                    initial_count = len(df)
                    df_clean = df.dropna()
                    removed_count = initial_count - len(df_clean)
                    
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} records with NaN values ({removed_count/initial_count*100:.1f}% of data)")
                    else:
                        logger.info("No NaN values found in the data")
                    
                    logger.info(f"Loaded {len(df_clean)} valid time points from {file_pattern}")
                    logger.info(f"Data period: {df_clean.index.min()} to {df_clean.index.max()}")
                    logger.info(f"SWH statistics - Mean: {df_clean['swh'].mean():.3f}m, "
                               f"Max: {df_clean['swh'].max():.3f}m, Min: {df_clean['swh'].min():.3f}m")
                    
                    return df_clean
                else:
                    raise ValueError("Required variables 'swh' or time coordinate ('time'/'valid_time') not found in data")
                    
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except FileNotFoundError:
            logger.error(f"No NetCDF files found with pattern: {file_pattern}")
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        except Exception as e:
            logger.error(f"Failed to load NetCDF files: {str(e)}")
            raise ValueError(f"Data loading failed: {str(e)}")
    
    def export_to_csv(self, data: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Export wave data to CSV format in the configured results directory.
        
        Parameters
        ----------
        data : pd.DataFrame
            Wave data to export
        filename : str, optional
            Output filename (default: from config)
            
        Returns
        -------
        str
            Path to the exported CSV file
            
        Raises
        ------
        ValueError
            If export fails
        """
        try:
            if filename is None:
                filename = self.config.csv_filename
            
            # Ensure .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Use output directory from config (typically "results")
            output_dir = Path(self.config.output_directory)
            
            # Create a "results" subdirectory if output_directory is not already "results"
            if output_dir.name.lower() != 'results':
                output_dir = output_dir / 'results'
            
            # Ensure directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            
            logger.info(f"Exporting data to CSV: {output_path}")
            
            # Debug: Print data info
            logger.info(f"Data shape: {data.shape}, columns: {list(data.columns)}")
            logger.info(f"Data index type: {type(data.index)}")
            logger.info(f"Sample data:\n{data.head()}")
            
            # Export using pandas directly (more reliable)
            logger.info(f"Exporting {len(data)} records to CSV using pandas")
            data.to_csv(output_path, index=True, header=True)
            logger.info(f"Pandas export completed to: {output_path}")
            
            # Verify export with detailed logging
            logger.info(f"Checking if file exists at: {output_path}")
            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                logger.info(f"CSV export successful: {file_size:.1f} KB saved to {output_path}")
                return str(output_path)
            else:
                # Additional debug info
                logger.error(f"CSV file was not created at expected path: {output_path}")
                logger.error(f"Directory contents: {list(output_dir.iterdir())}")
                raise ValueError(f"CSV file was not created at: {output_path}")
                
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            raise ValueError(f"Failed to export CSV: {str(e)}")
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean wave data by removing NaN values and invalid entries.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw wave data to clean
            
        Returns
        -------
        pd.DataFrame
            Cleaned wave data without NaN values
        """
        try:
            initial_count = len(data)
            logger.info(f"Starting data cleaning with {initial_count} records")
            
            # Remove NaN values
            data_clean = data.dropna()
            nan_removed = initial_count - len(data_clean)
            
            # Remove negative values (physically impossible for wave height)
            data_clean = data_clean[data_clean['swh'] >= 0]
            negative_removed = len(data_clean) - (initial_count - nan_removed)
            
            # Remove extremely high values (likely errors, >20m is very rare)
            data_clean = data_clean[data_clean['swh'] <= 20.0]
            outlier_removed = len(data_clean) - (initial_count - nan_removed + negative_removed)
            
            total_removed = initial_count - len(data_clean)
            
            logger.info(f"Data cleaning completed:")
            logger.info(f"  - NaN values removed: {nan_removed}")
            logger.info(f"  - Negative values removed: {abs(negative_removed)}")
            logger.info(f"  - Outliers (>20m) removed: {abs(outlier_removed)}")
            logger.info(f"  - Total records removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
            logger.info(f"  - Final clean dataset: {len(data_clean)} records")
            
            return data_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return data  # Return original data if cleaning fails
    

def download_era5_data(config: dict[str, Any]) -> dict[str, Any]:
    """Download and process ERA5 data from Climate Data Store.
    
    Main function that orchestrates the complete workflow for downloading and
    processing ERA5 data. Accepts a configuration dictionary and returns
    processing results including download statistics and output file paths.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - start_year (int): First year to download (required)
            - end_year (int): Last year to download (required)
            - area_bounds (list): Geographic bounds [North, West, South, East] (required)
            - output_directory (str): Directory for output files (required)
            - variable (str, optional): CDS variable name
            - file_prefix (str, optional): Prefix for output filenames
            - export_csv (bool, optional): Export data to CSV format (default: True)
    
    Returns:
        dict: Results dictionary containing:
            - config (ERA5DataDownloadConfig): Configuration object used
            - download_results (dict): Download status for each year
            - validation_results (dict): Validation status for each file
            - data (pd.DataFrame): Processed data
            - csv_path (str, optional): Path to exported CSV file
    
    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        ConnectionError: If CDS API client cannot be initialized.
        
    Example:
        >>> from environmentaltools.download import marine_copernicus
        >>> 
        >>> # Define configuration
        >>> config = {
        ...     'start_year': 2018,
        ...     'end_year': 2020,
        ...     'area_bounds': [41.4, -9.0, 41.0, -8.65],
        ...     'output_directory': './era5_data',
        ...     'variable': 'significant_height_of_combined_wind_waves_and_swell',
        ...     'export_csv': True
        ... }
        >>> 
        >>> # Download and process data
        >>> results = marine_copernicus.download_era5_data(config)
        >>> print(f"Downloaded {results['download_results']} years")
        >>> print(f"Data saved to: {results['csv_path']}")
    """
    try:
        # Initialize configuration from dictionary
        config_obj = ERA5DataDownloadConfig(config)
        
        logger.info("Starting ERA5 data download and processing workflow")
        
        # Print configuration summary
        config_obj.print_summary()
        
        # Initialize downloader
        downloader = ERA5DataDownloader(config_obj)
        
        # Download all years
        download_results = downloader.download_all_years()
        
        # Validate downloads
        validation_results = downloader.validate_downloaded_files()
        
        # Check if we have any valid files
        valid_files = [year for year, result in validation_results.items() if result.get('valid', False)]
        
        if not valid_files:
            logger.error("No valid NetCDF files available for processing")
            return {
                'config': config_obj,
                'download_results': download_results,
                'validation_results': validation_results,
                'data': None,
                'csv_path': None
            }
        
        logger.info(f"Proceeding with processing of {len(valid_files)} valid files")
        
        # Initialize processor
        processor = ERA5DataProcessor(config_obj)
        
        # Load and process data
        data = processor.load_netcdf_files()
        
        # Clean data (remove NaN and invalid values)
        data_clean = processor.clean_data(data)
        
        # Prepare results
        results = {
            'config': config_obj,
            'download_results': download_results,
            'validation_results': validation_results,
            'data': data_clean,
        }
        
        # Export to CSV if requested
        export_csv = config.get('export_csv', True)
        if export_csv:
            csv_path = processor.export_to_csv(data_clean)
            results['csv_path'] = csv_path
            logger.info(f"Data exported to CSV: {csv_path}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise
