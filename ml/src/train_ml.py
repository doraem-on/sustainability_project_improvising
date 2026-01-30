import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


df = pd.read_csv("data/processed/final_dataset.csv")



for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].nunique() == 2:  
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)


if 'efficiency' not in df.columns:
    raise ValueError("Target column 'efficiency' not found in dataset")

X = df.drop("efficiency", axis=1)
y = df["efficiency"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "ml/models/efficiency_model.pkl")
print("Model trained and saved successfully!")
