# ml/train_expert.py
import pandas as pd
import numpy as np  # Added numpy for sqrt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error

def train_system():
    print("Loading processed real-world data...")
    df = pd.read_csv("data/processed/training_data.csv")
    
    # --- MODEL 1: The "Digital Twin" (Regressor) ---
    # Predicts what AC POWER *should* be based on weather.
    feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER']
    X = df[feature_cols]
    y_reg = df['AC_POWER']
    
    print("Training Digital Twin (Regressor)...")
    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    regressor.fit(X, y_reg)
    
    # Evaluate Regressor (FIXED for Scikit-Learn 1.4+)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y_reg, y_pred)
    rmse = np.sqrt(mse) # Manual calculation is safer across versions
    print(f"Digital Twin RMSE: {rmse:.2f} kW (Avg error)")
    
    # --- MODEL 2: The "Fault Diagnostician" (Classifier) ---
    # Predicts the specific error category.
    y_cls = df['status']
    
    print("Training Fault Classifier...")
    classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    classifier.fit(X, y_cls)
    
    # Evaluate Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    print("Classifier Report:")
    print(classification_report(y_test, classifier.predict(X_test)))
    
    # Save Artifacts
    with open('ml/src/digital_twin_model.pkl', 'wb') as f:
        pickle.dump(regressor, f)
    with open('ml/src/fault_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
        
    print("Expert models saved successfully.")

if __name__ == "__main__":
    train_system()