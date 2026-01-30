import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------
# 1. Load processed dataset
# ---------------------------
df = pd.read_csv("data/processed/final_dataset.csv")

# ---------------------------
# 2. Handle categorical columns
# ---------------------------
# Convert Yes/No columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].nunique() == 2:  # Yes/No column
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            # One-hot encode other categorical columns like 'city'
            df = pd.get_dummies(df, columns=[col], drop_first=True)

# ---------------------------
# 3. Define features and target
# ---------------------------
if 'efficiency' not in df.columns:
    raise ValueError("Target column 'efficiency' not found in dataset")

X = df.drop("efficiency", axis=1)
y = df["efficiency"]

# ---------------------------
# 4. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Train model
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 6. Save trained model
# ---------------------------
joblib.dump(model, "ml/models/efficiency_model.pkl")
print("✅ Model trained and saved successfully!")
