import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")

df["power"] = df["voltage"] * df["current"]
df["temp_stress"] = df["panel_temp"] * df["temperature"]
df["weather_risk"] = df["humidity"] * df["dust_index"]

df.to_csv("data/processed/final_dataset.csv", index=False)
