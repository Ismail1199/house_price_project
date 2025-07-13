from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd

app = Flask(__name__)
# model = joblib.load('house_price_xgb_model.pkl')

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'house_price_xgb_model.pkl')
model = joblib.load(model_path)


FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built',
    'lat', 'long', 'sqft_living15', 'sqft_lot15',
    'house_age', 'was_renovated'
]

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/style.css')
def css():
    return send_file('style.css')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        input_df = input_df[FEATURES]  # Ensuring correct column order
        prediction = model.predict(input_df)[0]
        prediction = float(prediction)  # Ensuring it's a native Python float
        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(debug=True)
