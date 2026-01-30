import pandas as pd

sensor = pd.read_csv("data/synthetic/sensor_data.csv")
weather = pd.read_csv("data/raw/indian_weather_data.csv")

# Simplify for hackathon
weather = weather.sample(len(sensor), replace=True).reset_index(drop=True)

final_df = pd.concat([sensor, weather], axis=1)

final_df.dropna(inplace=True)
final_df.to_csv("data/processed/final_dataset.csv", index=False)
final_df.head()