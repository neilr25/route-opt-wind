#!/usr/bin/env python3
"""
Download ERA5 hourly 10m wind data and convert to Parquet format.

This script downloads ERA5 hourly wind component data (u10 and v10) from the Copernicus
Climate Data Store (CDS) for a specified month and geographic region, then converts
them to wind speed and direction and saves as Parquet files.

CDS API Setup:
--------------
1. Register at https://cds.climate.copernicus.eu/
2. Install cdsapi: pip install cdsapi
3. Configure your API key:
   - Create ~/.cdsapirc file with:
     url: https://cds.climate.copernicus.eu/api/v2
     key: YOUR_API_KEY_HERE
   - Or set CDS_API_KEY environment variable

Usage:
------
download_era5_hourly.py --year YYYY --month MM [--lat-min 29 --lat-max 61 --lon-min -81 --lon-max 11]

Examples:
---------
# Download January 2025 data with default region (North Atlantic corridor)
python download_era5_hourly.py --year 2025 --month 1

# Download June 2025 data with custom region
python download_era5_hourly.py --year 2025 --month 6 --lat-min 30 --lat-max 60 --lon-min -80 --lon-max 10

# Dry run - show request without downloading
python download_era5_hourly.py --year 2025 --month 6 --dry-run

Output:
-------
Files are saved to C:\\app\\data\\hourly\\weather_YYYY-MM_hourly.parquet
"""

import argparse
import os
import sys
import math
from datetime import datetime
from pathlib import Path

try:
    import cdsapi
except ImportError:
    cdsapi = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import xarray as xr
except ImportError:
    xr = None


def download_era5_data(year, month, lat_min, lat_max, lon_min, lon_max, dry_run=False):
    """
    Download ERA5 hourly wind data for specified region and time period.
    
    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
        lat_min: Minimum latitude
        lat_max: Maximum latitude  
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        dry_run: If True, show request without downloading
        
    Returns:
        Path to downloaded NetCDF file if successful, None otherwise
    """
    # Create output directories
    output_dir = Path("C:\\app\\data\\hourly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format month as two digits
    month_str = f"{month:02d}"
    
    # Output file name
    output_file = output_dir / f"weather_{year}-{month_str}_hourly.parquet"
    
    # Check if file already exists
    if output_file.exists():
        print(f"Error: File already exists: {output_file}")
        return None
    
    # Temporary NetCDF file
    temp_nc_file = output_dir / f"era5_{year}_{month_str}_temp.nc"
    
    try:
        if cdsapi is None:
            print("Error: cdsapi module not found. Please install with: pip install cdsapi")
            return None
            
        # Initialize CDS API client
        c = cdsapi.Client()
        
        # Prepare the request
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind'
            ],
            'year': str(year),
            'month': month_str,
            'day': [f'{day:02d}' for day in range(1, 32)],  # All days in month
            'time': [f'{hour:02d}:00' for hour in range(24)],  # All hours
            'area': [lat_max, lon_min, lat_min, lon_max],  # CDS uses [N, W, S, E] format
        }
        
        print(f"CDS API Request for {year}-{month_str}:")
        print(f"  Dataset: reanalysis-era5-single-levels")
        print(f"  Variables: 10m_u_component_of_wind, 10m_v_component_of_wind")
        print(f"  Region: [{lat_min}, {lon_min}] to [{lat_max}, {lon_max}]")
        print(f"  Time range: {year}-{month_str}-01 00:00 to {year}-{month_str}-31 23:00")
        print(f"  Output: {temp_nc_file}")
        
        if dry_run:
            print("\nDry run complete. No data downloaded.")
            return None
        
        print("\nDownloading data from CDS...")
        
        # Download the data
        c.retrieve(
            'reanalysis-era5-single-levels',
            request,
            str(temp_nc_file)
        )
        
        print(f"Download complete: {temp_nc_file}")
        return temp_nc_file
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        if 'API key' in str(e) or 'authentication' in str(e):
            print("\nPlease configure your CDS API key:")
            print("1. Create ~/.cdsapirc with your API key")
            print("2. Or set CDS_API_KEY environment variable")
            print("3. Register at https://cds.climate.copernicus.eu/")
        return None


def convert_to_parquet(nc_file, year, month):
    """
    Convert NetCDF data to Parquet format.
    
    Args:
        nc_file: Path to NetCDF file
        year: Year
        month: Month
        
    Returns:
        Path to Parquet file
    """
    if xr is None or np is None or pd is None:
        print("Error: Required modules not found. Please install with: pip install xarray numpy pandas pyarrow")
        return None
        
    print(f"Converting {nc_file} to Parquet...")
    
    # Load NetCDF data
    ds = xr.open_dataset(nc_file)
    
    # Extract wind components
    u10 = ds['u10']
    v10 = ds['v10']
    
    # Convert xarray to DataFrame using vectorized operations
    # Stack all dimensions into a flat table
    df = ds.to_dataframe().reset_index()

    # CDS API uses 'valid_time' instead of 'time' (new CDS API format)
    if 'valid_time' in df.columns:
        df = df.rename(columns={'valid_time': 'time'})
    
    # Convert u10/v10 to wind speed and direction using vectorized operations
    df['wind_speed_10m'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['wind_direction_10m'] = (270 - np.arctan2(df['v10'], df['u10']) * 180 / np.pi) % 360
    
    # Select only the columns we want
    df = df[['time', 'latitude', 'longitude', 'wind_speed_10m', 'wind_direction_10m']]
    
    # Convert to proper data types
    df['time'] = pd.to_datetime(df['time'])
    df['latitude'] = df['latitude'].astype('float64')
    df['longitude'] = df['longitude'].astype('float64')
    df['wind_speed_10m'] = df['wind_speed_10m'].astype('float64')
    df['wind_direction_10m'] = df['wind_direction_10m'].astype('float64')
    
    # Output file
    month_str = f"{month:02d}"
    output_file = Path("C:\\app\\data\\hourly") / f"weather_{year}-{month_str}_hourly.parquet"
    
    # Save as Parquet
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    
    print(f"Conversion complete: {output_file}")
    print(f"Data shape: {len(df)} rows")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
    print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")
    
    # Close xarray dataset handle before unlinking temp file
    ds.close()
    
    # Clean up temporary file
    try:
        nc_file.unlink()
    except Exception as cleanup_err:
        print(f"Warning: could not remove temp file {nc_file}: {cleanup_err}")
    
    return output_file


def main():
    """
    Main function to parse arguments and run the download/conversion process.
    """
    parser = argparse.ArgumentParser(
        description='Download ERA5 hourly 10m wind data and convert to Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_era5_hourly.py --year 2025 --month 6
  python download_era5_hourly.py --year 2025 --month 6 --lat-min 30 --lat-max 60 --lon-min -80 --lon-max 10
  python download_era5_hourly.py --year 2025 --month 6 --dry-run
        """
    )
    
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2025)')
    parser.add_argument('--month', type=int, required=True, choices=range(1, 13), help='Month (1-12)')
    parser.add_argument('--lat-min', type=float, default=29, help='Minimum latitude (default: 29)')
    parser.add_argument('--lat-max', type=float, default=61, help='Maximum latitude (default: 61)')
    parser.add_argument('--lon-min', type=float, default=-81, help='Minimum longitude (default: -81)')
    parser.add_argument('--lon-max', type=float, default=11, help='Maximum longitude (default: 11)')
    parser.add_argument('--dry-run', action='store_true', help='Show request without downloading')
    
    args = parser.parse_args()
    
    print(f"ERA5 Hourly Wind Data Downloader")
    print(f"Year: {args.year}, Month: {args.month}")
    print(f"Region: [{args.lat_min}, {args.lon_min}] to [{args.lat_max}, {args.lon_max}]")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Validate region
    if args.lat_min >= args.lat_max:
        print("Error: lat-min must be less than lat-max")
        sys.exit(1)
    
    if args.lon_min >= args.lon_max:
        print("Error: lon-min must be less than lon-max")
        sys.exit(1)
    
    # Check CDS API key configuration
    cds_config = Path.home() / '.cdsapirc'
    if not cds_config.exists() and not os.environ.get('CDS_API_KEY'):
        print("CDS API key not configured.")
        print("Please configure your CDS API key:")
        print("1. Create ~/.cdsapirc with your API key")
        print("2. Or set CDS_API_KEY environment variable")
        print("3. Register at https://cds.climate.copernicus.eu/")
        if not args.dry_run:
            sys.exit(1)
    
    # Download data
    nc_file = download_era5_data(
        args.year, args.month, args.lat_min, args.lat_max, 
        args.lon_min, args.lon_max, args.dry_run
    )
    
    if nc_file and not args.dry_run:
        # Convert to Parquet
        try:
            parquet_file = convert_to_parquet(nc_file, args.year, args.month)
            print(f"\nSuccess! Data saved to: {parquet_file}")
        except Exception as e:
            print(f"Error during conversion: {e}")
            # Clean up temporary file on error (try to close nc handle first)
            try:
                import xarray as xr_cleanup
                ds_cleanup = xr_cleanup.open_dataset(nc_file)
                ds_cleanup.close()
            except Exception:
                pass
            if nc_file.exists():
                try:
                    nc_file.unlink()
                    print(f"Cleaned up temporary file: {nc_file}")
                except PermissionError:
                    pass
            sys.exit(1)


if __name__ == '__main__':
    main()