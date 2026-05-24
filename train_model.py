import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# Uses sklearn's built-in California Housing dataset as a reliable alternative
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Dataset shape: {df.shape}")
print(df.describe())

# ── 2. EDA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of target
df['MedHouseVal'].hist(bins=50, ax=axes[0, 0], color='steelblue', edgecolor='white')
axes[0, 0].set_title('House Value Distribution')
axes[0, 0].set_xlabel('Median House Value ($100k)')

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', ax=axes[0, 1], cmap='coolwarm', linewidths=0.5)
axes[0, 1].set_title('Correlation Heatmap')

# Income vs Price
axes[1, 0].scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3, color='tomato', s=5)
axes[1, 0].set_xlabel('Median Income')
axes[1, 0].set_ylabel('House Value')
axes[1, 0].set_title('Income vs House Value')

# Rooms vs Price
axes[1, 1].scatter(df['AveRooms'], df['MedHouseVal'], alpha=0.3, color='seagreen', s=5)
axes[1, 1].set_xlabel('Average Rooms')
axes[1, 1].set_ylabel('House Value')
axes[1, 1].set_title('Avg Rooms vs House Value')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
plt.show()

# ── 3. Outlier Removal (IQR method) ──────────────────────────────────────────
Q1 = df['MedHouseVal'].quantile(0.25)
Q3 = df['MedHouseVal'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['MedHouseVal'] >= Q1 - 1.5 * IQR) & (df['MedHouseVal'] <= Q3 + 1.5 * IQR)]
print(f"\nAfter outlier removal: {df.shape}")

# ── 4. Feature Engineering ────────────────────────────────────────────────────
df['rooms_per_person']    = df['AveRooms'] / df['Population'].replace(0, np.nan)
df['bedrooms_per_room']   = df['AveBedrms'] / df['AveRooms'].replace(0, np.nan)
df['population_per_household'] = df['Population'] / df['HouseAge'].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude',
            'rooms_per_person', 'bedrooms_per_room', 'population_per_household']

X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 6. Scaling ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. Model Training ─────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── 8. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"\nRMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ── 9. Prediction Plot ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted  |  RMSE={rmse:.3f}  R²={r2:.3f}')
plt.tight_layout()
plt.savefig('prediction_plot.png', dpi=150)
plt.show()

# ── 10. Save Artefacts ────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)
print("Saved model, scaler, features.")
