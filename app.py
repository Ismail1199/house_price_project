from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('/house_price_project/house_price_xgb_model.pkl')

FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built',
    'lat', 'long', 'sqft_living15', 'sqft_lot15',
    'house_age', 'was_renovated'
]

@app.route('/')
def home():
    return send_file('/house_price_project/index.html')

@app.route('/house_price_project/style.css')
def css():
    return send_file('/house_price_project/style.css')

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
    app.run(debug=True)
