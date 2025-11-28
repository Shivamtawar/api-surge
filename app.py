# surge_prediction_api.py
"""
Flask API for Patient Surge Prediction System
Provides REST endpoints for disease surge predictions and resource planning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import os
import traceback

from MLmodel import (
    SurgePredictionModel,
    SurgePredictionEngine,
    Config
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global model instance
model = None
engine = None

# ==========================
# MODEL INITIALIZATION
# ==========================

def initialize_model():
    """Load the trained model on startup"""
    global model, engine
    
    try:
        model = SurgePredictionModel()
        
        if os.path.exists(Config.MODEL_SAVE_PATH):
            model.load_model()
            engine = SurgePredictionEngine(model)
            print("✅ Model loaded successfully!")
            return True
        else:
            print("⚠️  Model file not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


# ==========================
# HELPER FUNCTIONS
# ==========================

def validate_input(data, required_fields):
    """Validate that all required fields are present"""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, None


def format_prediction_response(results_df, summary, input_params):
    """Format the prediction results into a clean JSON response"""
    
    # Convert DataFrame to list of dicts
    disease_predictions = results_df.to_dict('records')
    
    # Format disease predictions
    formatted_diseases = []
    for pred in disease_predictions:
        disease_dict = {
            "disease": pred.get("Disease"),
            "predicted_cases": pred.get("Predicted_Cases"),
            "baseline_median": pred.get("Baseline_Median"),
            "surge_threshold": pred.get("Surge_Threshold"),
            "is_surge": pred.get("Surge_Flag"),
            "surge_status": pred.get("Is_Surge"),
            "resources": {
                "beds": pred.get("Beds_Needed"),
                "oxygen_units": pred.get("Oxygen_Units"),
                "ventilators": pred.get("Ventilators"),
                "ors_kits": pred.get("ORS_Kits"),
                "nebulizers": pred.get("Nebulizers"),
                "masks": pred.get("Masks"),
                "ppe_kits": pred.get("PPE_Kits"),
                "staff": pred.get("Staff_Required")
            }
        }
        
        # Add disease-specific resources
        specific_resources = {}
        for key, value in pred.items():
            if key not in ["Disease", "Predicted_Cases", "Baseline_Median", 
                          "Surge_Threshold", "Is_Surge", "Surge_Flag",
                          "Beds_Needed", "Oxygen_Units", "Ventilators",
                          "ORS_Kits", "Nebulizers", "Masks", "PPE_Kits", "Staff_Required"]:
                specific_resources[key.lower()] = value
        
        if specific_resources:
            disease_dict["disease_specific_resources"] = specific_resources
        
        formatted_diseases.append(disease_dict)
    
    # Format overall summary
    resource_summary = {
        "total_beds": summary.get("Total_Beds"),
        "total_oxygen_units": summary.get("Total_Oxygen_Units"),
        "total_ventilators": summary.get("Total_Ventilators"),
        "total_ors_kits": summary.get("Total_ORS_Kits"),
        "total_nebulizers": summary.get("Total_Nebulizer_Kits"),
        "total_masks": summary.get("Total_Masks"),
        "total_ppe_kits": summary.get("Total_PPE_Kits"),
        "total_staff": summary.get("Total_Staff_Required")
    }
    
    # Response structure
    response = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_parameters": input_params,
        "predictions": {
            "diseases": formatted_diseases,
            "summary": {
                "total_surges_detected": summary.get("Total_Surges_Detected"),
                "risk_level": summary.get("Risk_Level"),
                "resources_required": resource_summary,
                "advisories": summary.get("Advisories", [])
            }
        }
    }
    
    return response


# ==========================
# API ENDPOINTS
# ==========================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with documentation"""
    return jsonify({
        "service": "Patient Surge Prediction API",
        "version": "1.0.0",
        "status": "active" if engine is not None else "model not loaded",
        "endpoints": {
            "/health": "GET - Health check",
            "/api/predict": "POST - Predict disease surges and resources",
            "/api/predict/batch": "POST - Batch predictions for multiple scenarios",
            "/api/diseases": "GET - List available diseases",
            "/api/model/info": "GET - Model information and metrics"
        },
        "documentation": "https://github.com/your-repo/surge-prediction"
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if engine is not None else "unhealthy",
        "model_loaded": engine is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected JSON payload:
    {
        "city": "Delhi",
        "aqi": 420,
        "pm25": 320,
        "pm10": 450,
        "temperature": 22,
        "humidity": 40,
        "rainfall": 0.0,
        "season": "Autumn",
        "festival": "Diwali",
        "day_type": "Holiday",
        "city_population": 2000000,
        "diseases": ["Influenza", "Dengue"],  // Optional
        "surge_multiplier": 1.3  // Optional
    }
    """
    
    if engine is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first."
        }), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Required fields
        required_fields = [
            'city', 'aqi', 'pm25', 'pm10', 'temperature',
            'humidity', 'rainfall', 'season', 'festival', 'day_type'
        ]
        
        # Validate input
        is_valid, error_msg = validate_input(data, required_fields)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
        
        # Extract parameters
        params = {
            'city': data['city'],
            'aqi': float(data['aqi']),
            'pm25': float(data['pm25']),
            'pm10': float(data['pm10']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'rainfall': float(data['rainfall']),
            'season': data['season'],
            'festival': data.get('festival', 'None'),
            'day_type': data.get('day_type', 'Weekday'),
            'city_population': int(data.get('city_population', 1000000)),
            'diseases': data.get('diseases'),
            'surge_multiplier': float(data.get('surge_multiplier', Config.SURGE_MULTIPLIER))
        }
        
        # Make prediction
        results_df, summary = engine.predict_surge_and_resources(**params)
        
        # Format response
        response = format_prediction_response(results_df, summary, {
            'city': params['city'],
            'aqi': params['aqi'],
            'pm25': params['pm25'],
            'pm10': params['pm10'],
            'temperature': params['temperature'],
            'humidity': params['humidity'],
            'rainfall': params['rainfall'],
            'season': params['season'],
            'festival': params['festival'],
            'day_type': params['day_type'],
            'city_population': params['city_population']
        })
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid data type: {str(e)}"
        }), 400
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple scenarios
    
    Expected JSON payload:
    {
        "scenarios": [
            {
                "city": "Delhi",
                "aqi": 420,
                ...
            },
            {
                "city": "Mumbai",
                "aqi": 85,
                ...
            }
        ]
    }
    """
    
    if engine is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first."
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'scenarios' not in data:
            return jsonify({
                "success": False,
                "error": "No scenarios provided. Expected 'scenarios' array."
            }), 400
        
        scenarios = data['scenarios']
        
        if not isinstance(scenarios, list):
            return jsonify({
                "success": False,
                "error": "Scenarios must be an array"
            }), 400
        
        if len(scenarios) > 10:
            return jsonify({
                "success": False,
                "error": "Maximum 10 scenarios allowed per batch request"
            }), 400
        
        # Process each scenario
        results = []
        required_fields = [
            'city', 'aqi', 'pm25', 'pm10', 'temperature',
            'humidity', 'rainfall', 'season', 'festival', 'day_type'
        ]
        
        for idx, scenario in enumerate(scenarios):
            # Validate
            is_valid, error_msg = validate_input(scenario, required_fields)
            if not is_valid:
                results.append({
                    "scenario_index": idx,
                    "success": False,
                    "error": error_msg
                })
                continue
            
            # Extract parameters
            params = {
                'city': scenario['city'],
                'aqi': float(scenario['aqi']),
                'pm25': float(scenario['pm25']),
                'pm10': float(scenario['pm10']),
                'temperature': float(scenario['temperature']),
                'humidity': float(scenario['humidity']),
                'rainfall': float(scenario['rainfall']),
                'season': scenario['season'],
                'festival': scenario.get('festival', 'None'),
                'day_type': scenario.get('day_type', 'Weekday'),
                'city_population': int(scenario.get('city_population', 1000000)),
                'diseases': scenario.get('diseases'),
                'surge_multiplier': float(scenario.get('surge_multiplier', Config.SURGE_MULTIPLIER))
            }
            
            # Make prediction
            results_df, summary = engine.predict_surge_and_resources(**params)
            
            # Format response
            response = format_prediction_response(results_df, summary, {
                'city': params['city'],
                'aqi': params['aqi'],
                'season': params['season'],
                'day_type': params['day_type']
            })
            
            results.append({
                "scenario_index": idx,
                **response
            })
        
        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_scenarios": len(scenarios),
            "results": results
        }), 200
    
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Batch prediction failed: {str(e)}"
        }), 500


@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get list of diseases the model can predict"""
    
    if engine is None or model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 503
    
    try:
        diseases = list(model.median_baselines.keys())
        
        return jsonify({
            "success": True,
            "diseases": diseases,
            "total_count": len(diseases),
            "note": "Traffic_Accident predictions are automatically included"
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    
    if engine is None or model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 503
    
    try:
        info = {
            "success": True,
            "model_type": "Gradient Boosting Regressor",
            "features": {
                "numeric": model.NUM_FEATURES,
                "categorical": model.CAT_FEATURES,
                "total": len(model.NUM_FEATURES) + len(model.CAT_FEATURES)
            },
            "diseases": list(model.median_baselines.keys()),
            "baseline_medians": model.median_baselines,
            "configuration": {
                "surge_multiplier": Config.SURGE_MULTIPLIER,
                "n_estimators": Config.N_ESTIMATORS,
                "learning_rate": Config.LEARNING_RATE,
                "max_depth": Config.MAX_DEPTH
            },
            "traffic_factors": {
                "base_rate": Config.TRAFFIC_BASE_RATE,
                "weekend_multiplier": Config.WEEKEND_MULTIPLIER,
                "holiday_multiplier": Config.HOLIDAY_MULTIPLIER,
                "rain_multiplier": Config.RAIN_MULTIPLIER,
                "fog_multiplier": Config.FOG_MULTIPLIER
            }
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/example', methods=['GET'])
def get_example():
    """Get example request payload"""
    
    example_payload = {
        "city": "Delhi",
        "aqi": 420,
        "pm25": 320,
        "pm10": 450,
        "temperature": 22,
        "humidity": 40,
        "rainfall": 0.0,
        "season": "Autumn",
        "festival": "Diwali",
        "day_type": "Holiday",
        "city_population": 2000000,
        "diseases": ["Influenza", "Dengue", "Asthma"],
        "surge_multiplier": 1.3
    }
    
    return jsonify({
        "success": True,
        "example_request": example_payload,
        "notes": {
            "diseases": "Optional - defaults to all diseases",
            "surge_multiplier": "Optional - defaults to 1.3",
            "city_population": "Optional - defaults to 1,000,000",
            "festival": "Use 'None' if no festival",
            "day_type": "Options: Weekday, Saturday, Sunday, Holiday",
            "season": "Options: Summer, Monsoon, Autumn, Winter, Spring"
        }
    }), 200


# ==========================
# ERROR HANDLERS
# ==========================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "message": "Please check the correct HTTP method for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


# ==========================
# MAIN
# ==========================

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 PATIENT SURGE PREDICTION API")
    print("=" * 70)
    
    # Initialize model
    print("\n📦 Loading model...")
    model_loaded = initialize_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded. API will return errors.")
        print("   Please run 'python patient_surge_predictor.py' first to train the model.")
    
    print("\n" + "=" * 70)
    print("🚀 Starting Flask API Server...")
    print("=" * 70)
    print("\n📍 Endpoints:")
    print("   • GET  /                  - API documentation")
    print("   • GET  /health            - Health check")
    print("   • POST /api/predict       - Single prediction")
    print("   • POST /api/predict/batch - Batch predictions")
    print("   • GET  /api/diseases      - List diseases")
    print("   • GET  /api/model/info    - Model information")
    print("   • GET  /api/example       - Example request payload")
    print("\n" + "=" * 70)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )