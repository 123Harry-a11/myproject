# 📧 Spam Email Detection

NLP-based spam classifier using **TF-IDF Vectorization** and **Logistic Regression**.

## 🔧 Tech Stack
- Python, Scikit-learn, Pandas, Numpy
- TF-IDF Vectorizer, Logistic Regression
- Flask (backend API)
- HTML/CSS/JavaScript (frontend)
- Matplotlib (visualization)

## 📁 Project Structure
```
spam-detection/
├── train_model.py      # ML training script
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

### 2. Download Dataset
Download `spam.csv` from [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and place in root folder.

### 3. Train the model
```bash
python train_model.py
```

### 4. Run Flask app
```bash
python app.py
```
Visit `http://localhost:5000`

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~98% |
| Precision | ~97% |
| Recall | ~96% |

## 🧠 Techniques Used
- Text cleaning (lowercase, punctuation removal, digit removal)
- TF-IDF with bigrams (ngram_range 1-2)
- Logistic Regression with L2 regularization
- Confusion matrix & classification report visualization
