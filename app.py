from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        row = [float(data[f]) for f in features]
        scaled = scaler.transform([row])
        pred   = model.predict(scaled)[0]
        return jsonify({'predicted_price': round(pred * 100000, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
