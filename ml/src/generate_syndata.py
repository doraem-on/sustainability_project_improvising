import pandas as pd
import numpy as np

np.random.seed(42)

rows = 5000
data = {
    "panel_id": np.random.randint(1, 20, rows),
    "voltage": np.random.normal(35, 3, rows),
    "current": np.random.normal(8, 1, rows),
    "panel_temp": np.random.normal(45, 5, rows),
    "dust_index": np.random.uniform(0, 1, rows),
    "efficiency": np.random.normal(0.9, 0.05, rows)
}

df = pd.DataFrame(data)

# Simulate degradation
df.loc[df["dust_index"] > 0.7, "efficiency"] -= 0.15
df.loc[df["panel_temp"] > 55, "efficiency"] -= 0.1

df.to_csv("data/synthetic/sensor_data.csv", index=False)
