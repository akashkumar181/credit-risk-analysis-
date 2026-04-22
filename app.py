"""
Flask API for Credit Risk Model Deployment
Serves predictions via REST API endpoints
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/credit_risk_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Expected feature names (from training)
EXPECTED_FEATURES = [
    'duration', 'amount', 'installment_rate', 'present_residence', 'age',
    'number_credits', 'people_liable', 'status_... >= 200 DM / salary for at least 1 year',
    'status_0 <= ... < 200 DM', 'status_no checking account',
    'credit_history_critical account/other credits existing',
    'credit_history_delay in paying off in the past',
    'credit_history_existing credits paid back duly till now',
    'credit_history_no credits taken/all credits paid back duly',
    'purpose_car (new)', 'purpose_car (used)', 'purpose_domestic appliances',
    'purpose_education', 'purpose_furniture/equipment', 'purpose_others',
    'purpose_radio/television', 'purpose_repairs', 'purpose_retraining',
    'savings_... >= 1000 DM', 'savings_100 <= ... < 500 DM',
    'savings_500 <= ... < 1000 DM', 'savings_unknown/no savings account',
    'employment_duration_... >= 7 years', 'employment_duration_1 <= ... < 4 years',
    'employment_duration_4 <= ... < 7 years', 'employment_duration_unemployed',
    'personal_status_sex_male : divorced/separated',
    'personal_status_sex_male : married/widowed', 'personal_status_sex_male : single',
    'other_debtors_guarantor', 'other_debtors_none', 'property_car or other',
    'property_real estate', 'property_unknown/no property', 'other_installment_plans_none',
    'other_installment_plans_stores', 'housing_own', 'housing_rent',
    'job_skilled employee/official', 'job_unemployed/unskilled - non-resident',
    'job_unskilled - resident', 'telephone_yes', 'foreign_worker_yes'
]


@app.route('/', methods=['GET'])
def home():
    """Home endpoint - API information"""
    return jsonify({
        'status': 'success',
        'message': 'Credit Risk Model API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'This help message',
            'GET /health': 'Check API health',
            'POST /predict': 'Make prediction',
            'POST /batch_predict': 'Make batch predictions'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a single prediction
    
    Expected JSON format:
    {
        "features": [48 numerical values]
    }
    
    or
    
    {
        "data": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        # Handle array format
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        # Handle dictionary format
        elif 'data' in data:
            # Create dataframe with expected features
            input_data = data['data']
            features = pd.DataFrame([input_data])
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input format. Use "features" or "data" key.'
            }), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'prediction_label': 'DEFAULT RISK' if prediction == 1 else 'NO DEFAULT RISK',
            'probability_class_0': float(probability[0]),
            'probability_class_1': float(probability[1]),
            'confidence': float(max(probability))
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make batch predictions for multiple customers
    
    Expected JSON format:
    {
        "data": [
            [feat1, feat2, ...],
            [feat1, feat2, ...],
            ...
        ]
    }
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Expected "data" key with array of feature arrays'
            }), 400
        
        features = np.array(data['data'])
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'customer_id': i + 1,
                'prediction': int(pred),
                'prediction_label': 'DEFAULT RISK' if pred == 1 else 'NO DEFAULT RISK',
                'probability_class_0': float(prob[0]),
                'probability_class_1': float(prob[1]),
                'confidence': float(max(prob))
            })
        
        return jsonify({
            'status': 'success',
            'total_predictions': len(results),
            'predictions': results
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'status': 'success',
        'model_type': 'Logistic Regression',
        'training_accuracy': 0.965,
        'features_count': 48,
        'classes': ['No Default (0)', 'Default (1)'],
        'expected_features': EXPECTED_FEATURES[:10] + ['... (48 total)']
    }), 200


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Credit Risk Model API Server")
    print("=" * 50)
    print(f"Starting server at http://localhost:5000")
    print(f"API Documentation: http://localhost:5000/")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
