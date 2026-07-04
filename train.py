import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json

print("=" * 50)
print("HOUSE PRICE PREDICTION - TRAINING")
print("=" * 50)

# Load data
df = pd.read_csv('housing_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'rmse': rmse, 'r2': r2}
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

print("\n" + "=" * 50)
print("BEST MODEL:", max(results, key=lambda x: results[x]['r2']))

# Hyperparameter tuning for best model
print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.4f}")

# Save best model
best_model = grid_search.best_estimator_
with open('model.pkl', 'wb') as f:
    pickle.dump((best_model, scaler), f)

print("\n Model saved to model.pkl")

# Save metrics
metrics = {
    'best_model': 'GradientBoosting',
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'test_rmse': np.sqrt(mean_squared_error(y_test, best_model.predict(X_test_scaled))),
    'test_r2': r2_score(y_test, best_model.predict(X_test_scaled))
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(" Metrics saved to metrics.json")
