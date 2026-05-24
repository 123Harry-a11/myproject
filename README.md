# 🌿 Plant Disease Detection

CNN-based image classification model using TensorFlow/Keras for real-time plant disease detection.

## 🔧 Tech Stack
- Python, TensorFlow / Keras
- Custom CNN with BatchNorm, Dropout
- Flask (backend API)
- HTML/CSS/JavaScript (frontend)
- ImageDataGenerator for augmentation

## 📁 Project Structure
```
plant-disease-detection/
├── train_model.py               # CNN training script
├── app.py                       # Flask API
├── templates/
│   └── index.html               # Web interface (drag & drop upload)
├── dataset/                     # YOUR dataset here
│   ├── Healthy/
│   ├── Powdery_Mildew/
│   └── ...
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Recommended: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

Structure your `dataset/` folder as:
```
dataset/
├── Tomato_Healthy/
├── Tomato_Early_Blight/
├── Potato_Late_Blight/
└── ...
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Run Flask app
```bash
python app.py
```
Visit `http://localhost:5000` and upload a leaf image!

## 🧠 Techniques Used
- Data Augmentation (rotation, flip, zoom, shift)
- CNN with 3 Conv blocks + BatchNormalization
- EarlyStopping + ModelCheckpoint callbacks
- Softmax multi-class classification
- Real-time image upload and prediction via Flask
