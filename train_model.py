import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import re
import string

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# Using the classic SMS Spam Collection dataset
# Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# Place the file as 'spam.csv' in this directory
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
except FileNotFoundError:
    # Fallback: create a tiny demo dataset so the script still runs
    print("[INFO] spam.csv not found – using built-in demo data.")
    demo = {
        'label': ['ham','ham','spam','spam','ham','spam','ham','spam'],
        'message': [
            'Hey, are you free this evening?',
            'Meeting rescheduled to 3 pm tomorrow.',
            'Congratulations! You won a FREE iPhone. Click now!',
            'URGENT: Your account will be suspended. Verify NOW.',
            'Can you pick up some milk on the way home?',
            'Win cash prizes! Call this number immediately!',
            'See you at the gym tomorrow morning.',
            'You have been selected for a $1000 gift card. Claim now!'
        ]
    }
    df = pd.DataFrame(demo)

print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)                 # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()        # collapse whitespace
    return text

df['clean_message'] = df['message'].apply(preprocess)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

# ── 4. TF-IDF Vectorisation ───────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ── 5. Model Training ─────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
model.fit(X_train_tfidf, y_train)

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── 7. Visualisation ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# Label distribution
df['label'].value_counts().plot(kind='bar', ax=axes[1], color=['steelblue', 'tomato'],
                                 edgecolor='black')
axes[1].set_title('Label Distribution')
axes[1].set_xticklabels(['Ham', 'Spam'], rotation=0)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
print("Plot saved as results.png")

# ── 8. Save Artefacts ─────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("Model and vectoriser saved.")
