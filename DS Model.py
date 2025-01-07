import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load data (assuming CSV files have been processed and merged)
data = pd.read_csv('processed_data.csv', parse_dates=['DATE'], index_col='DATE')

# EDA - Visualizing trends
data.plot(subplots=True, figsize=(10, 12), title='Trends of Factors Affecting Home Prices')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Factors')
plt.show()

# Feature Engineering (Ensure all columns are numeric)
X = data[['FedFundsRate', 'UnemployedRate', 'CPI', 'HousingStarts']]
y = data['HomePriceIndex']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Model Evaluation
print("Linear Regression Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R^2 Score:", r2_score(y_test, y_pred_lr))

# Feature Importance (Linear Regression)
importance_lr = pd.Series(lr_model.coef_, index=X.columns).sort_values(ascending=False)
importance_lr.plot(kind='bar', title='Linear Regression Feature Importance')
plt.show()

# Random Forest Regressor (Alternative Model)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
print("\nRandom Forest Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))

# Feature Importance (Random Forest)
importance_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance_rf.plot(kind='bar', title='Random Forest Feature Importance')
plt.show()
