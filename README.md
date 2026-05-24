# 🏠 House Price Prediction

End-to-end Linear Regression model for predicting house prices with EDA and RMSE evaluation.

## 🔧 Tech Stack
- Python, Scikit-learn, Pandas, Numpy
- Linear Regression, StandardScaler
- Flask (backend API)
- HTML/CSS/JavaScript (frontend)
- Matplotlib, Seaborn (visualization)

## 📁 Project Structure
```
house-price-prediction/
├── train_model.py      # EDA + training script
├── app.py              # Flask API
├── templates/
│   └── index.html      # Web interface
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This auto-downloads the California Housing dataset via sklearn.

### 3. Run Flask app
```bash
python app.py
```
Visit `http://localhost:5000`

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| RMSE   | ~0.52 |
| R²     | ~0.63 |

## 🧠 Techniques Used
- Exploratory Data Analysis (EDA)
- IQR-based Outlier Removal
- Feature Engineering (rooms_per_person, bedrooms_per_room)
- StandardScaler normalization
- RMSE & R² evaluation metrics
- Correlation heatmap, scatter plots
