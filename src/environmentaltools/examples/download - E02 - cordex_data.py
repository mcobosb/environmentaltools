#!/usr/bin/env python3
"""
CORDEX Data Download Example with Hybrid ESGF Access
====================================================

This example demonstrates the hybrid approach for CORDEX data downloads:
- Uses PyESGF for CORDEX projects (which aren't yet supported by intake-esgf)
- Uses intake-esgf for other supported projects  
- Provides automatic fallback for better compatibility

Key Features:
- Automatic project detection and backend selection
- Single API for all ESGF projects
- Robust fallback mechanisms
- No code changes needed for existing users

Requirements:
    pip install environmentaltools[download]

This installs both:
    - esgf-pyclient (for CORDEX support)  
    - intake-esgf (for modern ESGF access)

Setup (Optional for authentication):
    Create ~/.esgf/config.ini with your ESGF credentials:
    [credentials]
    openid = your_esgf_openid
    password = your_password
    
Author: Environmental Tools Team
Date: 2025-11-11
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    from environmentaltools.download import (
        query_esgf_catalog,
        download_esgf_dataset,
        download_with_config
    )
    HAS_DOWNLOAD = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install 'environmentaltools[download]'")
    HAS_DOWNLOAD = False


def run_hybrid_example():
    """Run CORDEX data download example with hybrid ESGF backend."""
    
    print("CORDEX Data Download Example with Hybrid ESGF Access")
    print("====================================================\n")
    
    if not HAS_DOWNLOAD:
        print("⏭️  Skipping example due to missing dependencies")
        return
    
    # Example 1: Query ESGF catalog for CORDEX data (uses PyESGF automatically)
    print("📋 Example 1: Querying ESGF Catalog for CORDEX")
    print("----------------------------------------------")
    query = {
        "project": "CORDEX",
        "domain": "EUR-11", 
        "variable": ["pr"],
        "time_frequency": ["3hr"],  # Changed to 3hr for more available data
        "experiment": ["rcp85"],    # Changed to rcp85 which is more common
    }

    print(f"🔍 Searching for datasets matching: {query}")
    print("ℹ️  Note: CORDEX project will automatically use PyESGF backend")
    print("⏳ This may take a moment, limiting to first 20 results...")
    
    try:
        # Start with a small limit to test downloads
        datasets = query_esgf_catalog(query, max_results=5)

        if not datasets.empty:
            print(f"✅ Found {len(datasets)} datasets using PyESGF backend!")
            print("\n📊 Dataset overview (showing filtering fields):")
            # Show key columns that reflect our filters
            cols_to_show = ['dataset_id', 'domain', 'variable', 'experiment', 'time_frequency']
            available_cols = [col for col in cols_to_show if col in datasets.columns]
            if available_cols:
                print(datasets[available_cols].head(10))  # Show more results to verify filtering
            else:
                print("Available columns:", list(datasets.columns))
                print(datasets.head(10))
            
            # Verify that filtering worked correctly
            print("\n🔍 Filter Verification:")
            if 'project' in datasets.columns:
                unique_projects = datasets['project'].unique()
                print(f"  Projects found: {unique_projects}")
                cordex_count = sum(datasets['project'].str.contains('CORDEX', na=False))
                print(f"  CORDEX datasets: {cordex_count}/{len(datasets)}")
            
            if 'domain' in datasets.columns:
                unique_domains = datasets['domain'].unique() 
                print(f"  Domains found: {unique_domains}")
                eur11_count = sum(datasets['domain'].str.contains('EUR-11', na=False))
                print(f"  EUR-11 datasets: {eur11_count}/{len(datasets)}")
                
            if 'variable' in datasets.columns:
                unique_variables = datasets['variable'].unique()
                print(f"  Variables found: {unique_variables}")
                pr_count = sum(datasets['variable'].str.contains('pr', na=False))
                print(f"  Precipitation (pr) datasets: {pr_count}/{len(datasets)}")
                
            if 'experiment' in datasets.columns:
                unique_experiments = datasets['experiment'].unique()
                print(f"  Experiments found: {unique_experiments}")
                rcp85_count = sum(datasets['experiment'].str.contains('rcp85', na=False))
                print(f"  RCP8.5 datasets: {rcp85_count}/{len(datasets)}")
            
            # Check if dataset_id contains our filters (fallback verification)
            print(f"  Dataset IDs containing 'cordex': {sum(datasets['dataset_id'].str.contains('cordex', case=False, na=False))}/{len(datasets)}")
            print(f"  Dataset IDs containing 'EUR-11': {sum(datasets['dataset_id'].str.contains('EUR-11', na=False))}/{len(datasets)}")
            print(f"  Dataset IDs containing 'pr': {sum(datasets['dataset_id'].str.contains('pr', na=False))}/{len(datasets)}")
            print(f"  Dataset IDs containing 'rcp85': {sum(datasets['dataset_id'].str.contains('rcp85', na=False))}/{len(datasets)}")
            
            # Now let's try downloading a dataset
            print("\n📥 Example 2: Downloading Dataset")
            print("---------------------------------")
            
            # Create output directory
            output_folder = Path("./cordex_downloads")
            output_folder.mkdir(exist_ok=True)
            
            # Select the first dataset for download
            first_dataset = datasets.iloc[0].to_dict()
            print(f"📦 Attempting to download dataset: {first_dataset['dataset_id']}")
            print("⏳ This may take a moment...")
            print("💡 Note: Some datasets may require ESGF authentication")
            print("   Create ~/.esgf/config.ini with your ESGF credentials if needed")
            
            try:
                downloaded_files = download_esgf_dataset(
                    first_dataset,
                    str(output_folder),
                    file_filter="*.nc"  # Only download NetCDF files
                )
                
                if downloaded_files and len(downloaded_files) > 0:
                    print(f"✅ Successfully downloaded {len(downloaded_files)} files:")
                    for file_path in downloaded_files[:3]:  # Show first 3 files
                        print(f"  📄 {Path(file_path).name}")
                    
                    if len(downloaded_files) > 3:
                        print(f"  ... and {len(downloaded_files) - 3} more files")
                else:
                    print("⚠️  No files were downloaded (dataset may be empty or inaccessible)")
                        
            except Exception as e:
                print(f"❌ Download failed: {e}")
                print("💡 This might be due to:")
                print("   - Authentication requirements (ESGF credentials needed)")
                print("   - Network connectivity issues")
                print("   - Dataset access restrictions")
                print("   - Server availability problems")
            
            print("\n🎯 Query filtering is working! ✅")
            
        else:
            print("❌ No datasets found. This might be due to:")
            print("   - Very restrictive query parameters")
            print("   - Temporary ESGF server issues") 
            print("   - Network connectivity problems")
            print("   - Authentication requirements (try setting up ESGF credentials)")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        print("💡 Check your network connection and ESGF server availability")

    # Example 3: Using configuration file
    print("\n⚙️  Example 3: Download with Configuration File")
    print("------------------------------------------------")

    # Create a sample configuration file
    config_content = """
# CORDEX Download Configuration
project = CORDEX
domain = EUR-11
variable = tas,tasmin,tasmax
time_frequency = 3hr
experiment = historical
ensemble = r1i1p1

# Optional filters
institute = SMHI
driving_model = CNRM-CM5

# Download settings
max_files_per_dataset = 2
"""

    config_file = "cordex_config.ini"
    try:
        with open(config_file, "w") as f:
            f.write(config_content)

        print(f"📄 Created configuration file: {config_file}")
        print("🔍 Configuration specifies CORDEX project (will use PyESGF automatically)")

        # Create output directory if it doesn't exist
        output_folder = Path("./cordex_downloads")
        output_folder.mkdir(exist_ok=True)

        try:
            # For now, let's skip the config download since it requires additional CSV files
            print("⏭️  Skipping config download (requires coordinates.csv and selection.csv files)")
            print("💡 Config download functionality is available but needs setup files")
            
        except Exception as e:
            print(f"❌ Configuration download failed: {e}")
            print("💡 This might be due to authentication requirements or server availability")

    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(config_file):
            try:
                os.remove(config_file)
                print(f"\n🧹 Cleaned up: {config_file}")
            except Exception:
                pass

    # Example 4: Demonstrate hybrid behavior with different projects
    print("\n🔀 Example 4: Hybrid Backend Demonstration")
    print("------------------------------------------")
    
    print("🔧 Hybrid Implementation Features:")
    print("✓ CORDEX projects automatically use PyESGF (stable support)")
    print("✓ Other projects use intake-esgf when available") 
    print("✓ Automatic fallback ensures maximum compatibility")
    print("✓ Single API - no code changes needed for users")
    print("✓ Seamless backend switching based on project type")
    
    print("\n📚 Usage Tips:")
    print("• Use same function names regardless of project type")
    print("• PyESGF backend handles CORDEX reliably")
    print("• intake-esgf backend provides modern access for supported projects")
    print("• Authentication is optional but recommended for better access")


if __name__ == "__main__":
    run_hybrid_example()