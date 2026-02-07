"""
Solar Panel Dataset Merger
Combines sensor, weather, and site data into a single comprehensive dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_datasets():
    """Load all three datasets"""
    print("Loading datasets...")
    
    df_sensor = pd.read_csv('data/synthetic/sensor_data.csv')
    df_weather = pd.read_csv('data/raw/indian_weather_data.csv')
    df_sites = pd.read_csv('data/raw/Solar_Sites_Dataset_India.csv')
    
    print(f"Sensor data: {df_sensor.shape}")
    print(f"Weather data: {df_weather.shape}")
    print(f"Sites data: {df_sites.shape}")
    
    return df_sensor, df_weather, df_sites

def merge_datasets(df_sensor, df_weather, df_sites):
    """
    Merge the three datasets into one comprehensive dataset
    """
    print("\nMerging datasets...")
    
    # Step 1: Sample weather features for each sensor row
    # This randomly assigns weather conditions to each sensor reading
    weather_sample = df_weather.sample(len(df_sensor), replace=True, random_state=42).reset_index(drop=True)
    
    # Extract relevant weather columns
    weather_cols = ['temperature', 'humidity', 'cloudcover', 'precip', 'wind_speed']
    # Only use columns that exist in the weather dataset
    available_weather_cols = [col for col in weather_cols if col in df_weather.columns]
    
    # Step 2: Concatenate sensor data with weather features
    df_merged = pd.concat([
        df_sensor.reset_index(drop=True), 
        weather_sample[available_weather_cols]
    ], axis=1)
    
    # Step 3: Add site information (if site_id exists in sensor data, otherwise randomly assign)
    if 'site_id' in df_sensor.columns and 'site_id' in df_sites.columns:
        # Merge based on site_id
        df_merged = df_merged.merge(df_sites, on='site_id', how='left', suffixes=('', '_site'))
    else:
        # Randomly sample site data for each row
        sites_sample = df_sites.sample(len(df_merged), replace=True, random_state=42).reset_index(drop=True)
        df_merged = pd.concat([df_merged, sites_sample], axis=1)
    
    return df_merged

def add_synthetic_features(df_merged):
    """
    Add synthetic features and derive calculated fields
    """
    print("\nAdding synthetic and derived features...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Add synthetic irradiance (W/m²) if not present
    if 'irradiance' not in df_merged.columns:
        # Realistic irradiance values: 0-1200 W/m²
        # Higher during day, influenced by cloud cover if available
        df_merged['irradiance'] = np.random.uniform(0, 1200, len(df_merged))
        
        # Adjust based on cloud cover if available
        if 'cloudcover' in df_merged.columns:
            # More clouds = less irradiance
            df_merged['irradiance'] = df_merged['irradiance'] * (1 - df_merged['cloudcover'] / 100 * 0.7)
    
    # Calculate panel temperature
    # Panel temp = ambient temp + heating from irradiance
    if 'panel_temp' not in df_merged.columns:
        if 'temperature' in df_merged.columns:
            # Panel heats up more with higher irradiance
            df_merged['panel_temp'] = df_merged['temperature'] + (df_merged['irradiance'] / 800.0) * 20
        else:
            # Default panel temp if no ambient temp available
            df_merged['panel_temp'] = 25 + (df_merged['irradiance'] / 800.0) * 20
    
    # Adjust efficiency based on environmental factors
    if 'efficiency' in df_merged.columns:
        print("Adjusting efficiency based on environmental factors...")
        
        # Create base efficiency if needed
        base_efficiency = df_merged['efficiency'].copy()
        
        # Factor 1: Irradiance effect (more light = better, but normalize to 1000 W/m² standard)
        irradiance_factor = (df_merged['irradiance'] / 1000.0).clip(0, 1.2)
        
        # Factor 2: Temperature effect (efficiency drops ~0.5% per degree above 25°C)
        if 'temperature' in df_merged.columns:
            temp_factor = 1 - 0.005 * (df_merged['temperature'] - 25)
        else:
            temp_factor = 1
        
        # Factor 3: Humidity effect (slight reduction with high humidity)
        if 'humidity' in df_merged.columns:
            humidity_factor = 1 - 0.001 * df_merged['humidity']
        else:
            humidity_factor = 1
        
        # Factor 4: Precipitation effect (rain reduces efficiency)
        if 'precip' in df_merged.columns:
            precip_factor = 1 - 0.002 * df_merged['precip']
        else:
            precip_factor = 1
        
        # Apply all factors
        df_merged['efficiency'] = (base_efficiency * 
                                   irradiance_factor * 
                                   temp_factor * 
                                   humidity_factor * 
                                   precip_factor).clip(0, 1)
    
    # Add power output if capacity is available
    if 'capacity' in df_merged.columns or 'installed_capacity' in df_merged.columns:
        capacity_col = 'capacity' if 'capacity' in df_merged.columns else 'installed_capacity'
        df_merged['power_output'] = (df_merged[capacity_col] * 
                                     df_merged['efficiency'] * 
                                     (df_merged['irradiance'] / 1000.0))
    
    return df_merged

def clean_dataset(df):
    """
    Clean the merged dataset
    """
    print("\nCleaning dataset...")
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Ensure efficiency is between 0 and 1
    if 'efficiency' in df.columns:
        df['efficiency'] = df['efficiency'].clip(0, 1)
    
    # Ensure non-negative values for physical quantities
    for col in ['irradiance', 'temperature', 'humidity', 'wind_speed', 'power_output']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    return df

def save_dataset(df, output_path):
    """
    Save the merged dataset
    """
    print(f"\nSaving merged dataset to {output_path}...")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved successfully!")
    print(f"Final shape: {df.shape}")
    print(f"\nColumns in final dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    return output_path

def generate_summary(df):
    """
    Generate summary statistics for the merged dataset
    """
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal records: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    
    print("\nNumeric Features Summary:")
    print(df.describe().round(2))
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    return df

def main():
    """
    Main execution function
    """
    print("="*80)
    print("SOLAR PANEL DATASET MERGER")
    print("="*80)
    
    try:
        # Load datasets
        df_sensor, df_weather, df_sites = load_datasets()
        
        # Merge datasets
        df_merged = merge_datasets(df_sensor, df_weather, df_sites)
        
        # Add synthetic and derived features
        df_merged = add_synthetic_features(df_merged)
        
        # Clean dataset
        df_merged = clean_dataset(df_merged)
        
        # Save dataset
        output_path = 'data/processed/solar_panel_combined_dataset.csv'
        save_dataset(df_merged, output_path)
        
        # Generate summary
        generate_summary(df_merged)
        
        print("\n" + "="*80)
        print("MERGE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return df_merged
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find file - {e}")
        print("\nPlease ensure the following files exist:")
        print("  - data/synthetic/sensor_data.csv")
        print("  - data/raw/indian_weather_data.csv")
        print("  - data/raw/Solar_Sites_Dataset_India.csv")
        return None
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df_combined = main()
