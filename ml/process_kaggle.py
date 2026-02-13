# ml/process_kaggle.py
import pandas as pd
import numpy as np

def load_and_process_data():
    print("Loading Kaggle datasets...")
    # 1. Load Data
    gen_df = pd.read_csv("data/raw/Plant_1_Generation_Data.csv")
    weather_df = pd.read_csv("data/raw/Plant_1_Weather_Sensor_Data.csv")
    
    # 2. Fix Datetime Formats (Crucial step often missed)
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    
    # 3. Merge Weather into Generation
    # Weather is per plant, Generation is per Inverter. We merge on Time.
    df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
    
    # Drop redundant columns
    df = df.drop(columns=['PLANT_ID_x', 'PLANT_ID_y', 'SOURCE_KEY_y'])
    df = df.rename(columns={'SOURCE_KEY_x': 'inverter_id'})
    
    # 4. Feature Engineering (The "Expert" Touch)
    # Calculate Conversion Efficiency (AC / DC)
    # Avoid division by zero
    df['conversion_efficiency'] = np.where(df['DC_POWER'] > 0, df['AC_POWER'] / df['DC_POWER'], 0)
    
    # Create "Physics Lag" features (How panel responded 15 mins ago)
    df = df.sort_values(['inverter_id', 'DATE_TIME'])
    df['prev_ac'] = df.groupby('inverter_id')['AC_POWER'].shift(1)
    df['prev_temp'] = df.groupby('inverter_id')['MODULE_TEMPERATURE'].shift(1)
    df.fillna(0, inplace=True)
    
    # 5. GENERATE "FAULT" LABELS (Physics-Guided Labeling)
    # We define a fault as: High Irradiance + Low Output
    
    # Calculate expected output ratio (AC Power / Irradiance)
    # Normalizing to handle different times of day
    df['performance_ratio'] = np.where(df['IRRADIATION'] > 0, df['AC_POWER'] / df['IRRADIATION'], 0)
    
    # Define Thresholds based on data distribution (e.g., bottom 5% performance is a fault)
    # But only consider times when sun is actually shining (Irradiance > 0.1)
    daytime = df[df['IRRADIATION'] > 0.1]
    threshold = daytime['performance_ratio'].quantile(0.05) # Bottom 5% are "Underperforming"
    
    def classify_status(row):
        if row['IRRADIATION'] < 0.1:
            return "Night/Low_Light"
        elif row['performance_ratio'] < threshold:
            # Differentiate faults based on other vars
            if row['MODULE_TEMPERATURE'] > 60:
                return "Overheating"
            elif row['conversion_efficiency'] < 0.90: # Inverter should be ~97% efficient
                return "Inverter_Efficiency_Loss"
            else:
                return "Panel_Cleaning_Required" # Catch-all for low output
        else:
            return "Normal"

    df['status'] = df.apply(classify_status, axis=1)
    
    # Filter out Night data for training (Models get confused by 0s)
    clean_df = df[df['status'] != "Night/Low_Light"]
    
    print(f"Processed {len(clean_df)} records.")
    print("Fault Distribution:\n", clean_df['status'].value_counts())
    
    clean_df.to_csv("data/processed/training_data.csv", index=False)
    return clean_df

if __name__ == "__main__":
    load_and_process_data()