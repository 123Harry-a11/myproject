import os, json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('plant_disease_model.keras')
with open('class_indices.json') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = (128, 128)

def predict_image(img_path):
    img  = image.load_img(img_path, target_size=IMG_SIZE)
    arr  = image.img_to_array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    idx  = int(np.argmax(pred))
    return idx_to_class[idx], round(float(pred[idx]) * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(path)
    label, confidence = predict_image(path)
    os.remove(path)
    return jsonify({'disease': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
