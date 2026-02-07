import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import xgboost as xgb
import joblib 

df  = pd.read_csv('solar_panel_combined_dataset.csv')
df['Label'] = df['Label (Yes/No)'].map({'Yes':1,'No':0})
df.drop('panel_id', axis=1, inplace=True)
for col in df.columns:
    if df[col].dtype == 'object':
        df = pd.get_dummies(df, columns=[col], drop_first=True)
X_eff = df.drop(['efficiency', 'Label'], axis=1)  
y_eff = df['efficiency']
X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(X_eff, y_eff, test_size=0.2, random_state=42)
#efficieny one
model_eff = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror')
model_eff.fit(X_train_eff, y_train_eff)
y_pred_eff = model_eff.predict(X_test_eff)
r2_eff = r2_score(y_test_eff, y_pred_eff)
mae_eff = mean_absolute_error(y_test_eff, y_pred_eff)
print(f"Efficiency Model - R²: {r2_eff:.4f} (Target >0.94), MAE: {mae_eff:.4f}")
#sustainabilty one
X_site = df.drop(['efficiency', 'Label'], axis=1)  
y_site = df['Label']
X_train_site, X_test_site, y_train_site, y_test_site = train_test_split(X_site, y_site, test_size=0.2, random_state=42)

model_site = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model_site.fit(X_train_site, y_train_site)

y_pred_site = model_site.predict(X_test_site)
acc_site = accuracy_score(y_test_site, y_pred_site)
print(f"Suitability Model Accuracy: {acc_site:.4f}")

#save for first
joblib.dump(model_eff, "efficiency_model.pkl")
joblib.dump(X_eff.columns.tolist(), "feature_columns.pkl")
#save for 2nd
joblib.dump(model_site, "suitability_model.pkl")
joblib.dump(X_site.columns.tolist(), "site_feature_columns.pkl")
print("="*100)
print("all pkl models done")
print("="*100)
