from flask import Flask, request, jsonify, render_template
import pickle
import re
import string

app = Flask(__name__)

# Load saved model & vectoriser
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message.strip():
        return jsonify({'error': 'Empty message'}), 400

    clean = preprocess(message)
    vec   = tfidf.transform([clean])
    pred  = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0]

    return jsonify({
        'prediction': 'Spam' if pred == 1 else 'Ham',
        'confidence': round(float(max(prob)) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
