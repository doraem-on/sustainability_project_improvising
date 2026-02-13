# ml/src/app.py - THE COMPLETE SUPER APP
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import pickle
import traceback

# --- 1. CONFIGURATION & PATHS ---
# Get the directory where THIS file (app.py) lives: ml/src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Data Directory (Two levels up: ml/src -> ml -> root -> data)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))

# Define Static Directory (Inside ml/src/static)
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=STATIC_DIR)
CORS(app)

print(f"📂 App Location:  {BASE_DIR}")
print(f"📂 Data Location: {DATA_DIR}")

# --- 2. LOAD LEGACY MODELS (Site Planning) ---
print("\n--- LOADING LEGACY MODELS ---")
USE_OPENAI = False 
OPENAI_API_KEY = "api"

try:
    # Models are right next to app.py in ml/src
    model_eff = joblib.load(os.path.join(BASE_DIR, "efficiency_model.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    model_site = joblib.load(os.path.join(BASE_DIR, "suitability_model.pkl"))
    site_feature_columns = joblib.load(os.path.join(BASE_DIR, "site_feature_columns.pkl"))
    LEGACY_READY = True
    print("✅ Legacy Models (Efficiency & Suitability) loaded successfully")
except FileNotFoundError as e:
    LEGACY_READY = False
    print(f"⚠️ Legacy Models NOT found in {BASE_DIR}: {e}")

# Load SHAP (Legacy)
try:
    import sys
    sys.path.append(BASE_DIR) # Add ml/src to path so we can import 'explain'
    from explain import explain_prediction
    SHAP_AVAILABLE = True
    print("✅ SHAP explanation module loaded")
except ImportError:
    SHAP_AVAILABLE = False
    print("ℹ️ SHAP not available - using simplified explanations")
    
    def explain_prediction(data):
        return {
            "temperature": -0.05 if data.get('temperature', 25) > 35 else 0.02,
            "humidity": -0.03 if data.get('humidity', 50) > 70 else 0.01,
            "irradiance": 0.08 if data.get('irradiance', 800) > 700 else -0.05,
            "dust_index": -0.06 if data.get('dust_index', 0.5) > 0.5 else 0.01
        }

# Load OpenAI (Legacy)
if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    except ImportError:
        print("ℹ️ OpenAI library not installed.")
        USE_OPENAI = False
    except Exception:
        USE_OPENAI = False

# --- 3. LOAD ENTERPRISE MODELS (Predictive Maintenance) ---
print("\n--- LOADING ENTERPRISE MODELS ---")
try:
    with open(os.path.join(BASE_DIR, 'digital_twin_model.pkl'), 'rb') as f:
        digital_twin = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'fault_classifier.pkl'), 'rb') as f:
        fault_classifier = pickle.load(f)
    ENTERPRISE_READY = True
    print("✅ Enterprise Models loaded successfully")
except FileNotFoundError:
    ENTERPRISE_READY = False
    print(f"❌ Enterprise Models NOT found in {BASE_DIR}")

# --- 4. HELPER FUNCTIONS ---

def get_fallback_insights(shap_explanation, efficiency):
    """Restored Rule-based insights logic"""
    insights = []
    if efficiency > 0.85:
        insights.append("✓ Excellent performance - System is operating at peak efficiency.")
    elif efficiency > 0.75:
        insights.append("⚠ Good performance with room for optimization.")
    else:
        insights.append("⚠ Low efficiency detected - Immediate attention required.")
    
    if isinstance(shap_explanation, dict):
        top_factors = sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, impact in top_factors:
            if 'temperature' in feature.lower() and impact < -0.03:
                insights.append("🌡️ High temperature is reducing efficiency.")
            elif 'irradiance' in feature.lower() and impact > 0.05:
                insights.append("☀️ Good solar irradiance levels.")
            elif 'dust' in feature.lower() and impact < -0.03:
                insights.append("🧹 Dust accumulation detected. Schedule cleaning.")
            elif 'humidity' in feature.lower() and impact < -0.02:
                insights.append("💧 High humidity affecting performance.")
    
    if efficiency < 0.80:
        insights.append("📊 Recommendation: Conduct full system diagnostic.")
    else:
        insights.append("🔄 Recommendation: Continue regular maintenance.")
    return "\n".join(insights)

def get_genai_insights(shap_explanation, efficiency):
    """OpenAI wrapper"""
    if USE_OPENAI:
        try:
            pass 
        except Exception:
            pass
    return get_fallback_insights(shap_explanation, efficiency)

# --- 5. ROUTES ---

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(STATIC_DIR, 'main.html')

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "legacy": LEGACY_READY,
        "enterprise": ENTERPRISE_READY
    })

# --- LEGACY ROUTE (Restored) ---
@app.route("/predict", methods=["POST"])
def predict():
    if not LEGACY_READY: return jsonify({"error": "Legacy models missing"}), 503
    try:
        data = request.json
        # Feature Engineering
        data['panel_temp'] = data.get('temperature', 25) + (data.get('irradiance', 0) / 800.0) * 20
        defaults = {'cloudcover':0, 'precip':0, 'wind_speed':5, 'voltage':35, 'current':8, 'dust_index':0.5}
        for k, v in defaults.items(): 
            if k not in data: data[k] = v
        
        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        efficiency = float(model_eff.predict(df)[0])
        efficiency = max(0.0, min(1.0, efficiency))
        risk_score = round((1 - efficiency) * 100, 2)
        
        # Explanations
        if SHAP_AVAILABLE:
            try:
                explanation = explain_prediction(data)
            except:
                explanation = {"note": "Simulated Explanation"}
        else:
            explanation = {"note": "Simulated Explanation"}
            
        insights = get_genai_insights(explanation, efficiency)
        
        # Suitability
        suitability = "Yes" if efficiency > 0.7 else "No"

        return jsonify({
            "predicted_efficiency": round(efficiency, 3),
            "risk_score": risk_score,
            "failure_flag": efficiency < 0.75,
            "explanation": explanation,
            "insights_and_suggestions": insights,
            "recommended_action": "Optimize" if risk_score < 60 else "Immediate Action",
            "suitability": suitability
        })
    except Exception as e:
        print(f"Legacy Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- ENTERPRISE ROUTE (New) ---
@app.route('/api/analyze_telemetry', methods=['POST'])
def analyze_telemetry():
    if not ENTERPRISE_READY: return jsonify({"error": "Enterprise models missing"}), 503
    try:
        data = request.json
        input_data = pd.DataFrame([{
            'AMBIENT_TEMPERATURE': float(data.get('ambient_temp', 25)),
            'MODULE_TEMPERATURE': float(data.get('module_temp', 45)),
            'IRRADIATION': float(data.get('irradiance', 0)),
            'DC_POWER': float(data.get('dc_power', 0))
        }])

        expected = digital_twin.predict(input_data)[0]
        actual = float(data.get('ac_power', 0))
        health = min(100, max(0, (actual/expected)*100)) if expected > 0.1 else 100
        
        status = fault_classifier.predict(input_data)[0]
        prob = fault_classifier.predict_proba(input_data).max() * 100
        
        if status == "Normal" and health < 70: status = "Unknown Efficiency Drop"

        return jsonify({
            'status': 'success',
            'digital_twin_analysis': {'expected_power_kw': round(expected, 2), 'health_index': round(health, 1)},
            'diagnosis': {'condition': status, 'confidence_percent': round(prob, 1)}
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulation/next_reading', methods=['GET'])
def get_simulation_data():
    try:
        # Load from DATA_DIR (../../data)
        csv_path = os.path.join(DATA_DIR, 'processed', 'training_data.csv')
        
        if not os.path.exists(csv_path):
             return jsonify({'status': 'error', 'message': f'Data not found at {csv_path}'})
        
        df = pd.read_csv(csv_path).sample(1)
        record = df.iloc[0]
        return jsonify({
            'ambient_temp': float(record['AMBIENT_TEMPERATURE']),
            'module_temp': float(record['MODULE_TEMPERATURE']),
            'irradiance': float(record['IRRADIATION']),
            'dc_power': float(record['DC_POWER']),
            'ac_power': float(record['AC_POWER']),
            'timestamp': str(record['DATE_TIME'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    print(f"🚀 Server running on http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')