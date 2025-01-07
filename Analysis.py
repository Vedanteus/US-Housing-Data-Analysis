import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV data
cs_index = pd.read_csv('case_shiller_index.csv', index_col='DATE', parse_dates=True)
fed_funds_rate = pd.read_csv('fed_funds_rate.csv', index_col='DATE', parse_dates=True)
unemployment_rate = pd.read_csv('processed_unemployment_rate.csv', index_col='DATE', parse_dates=True)
cpi_data = pd.read_csv('cpi_data.csv', index_col='DATE', parse_dates=True)
housing_starts = pd.read_csv('housing_starts.csv', index_col='DATE', parse_dates=True)

# Resample to quarterly frequency using 'QE' instead of 'Q'
cs_index = cs_index.resample('QE').mean()
fed_funds_rate = fed_funds_rate.resample('QE').mean()
cpi_data = cpi_data.resample('QE').mean()
housing_starts = housing_starts.resample('QE').mean()
unemployment_rate = unemployment_rate.resample('QE').mean()


# Merge datasets into a single DataFrame
data = cs_index.rename(columns={'CSUSHPISA': 'HomePriceIndex'}).join([
    fed_funds_rate.rename(columns={'FEDFUNDS': 'FedFundsRate'}),
    unemployment_rate.rename(columns={'UnemploymentRate': 'UnemployedRate'}),
    cpi_data.rename(columns={'CPIAUCSL': 'CPI'}),
    housing_starts.rename(columns={'HOUST': 'HousingStarts'})
])

# Handle missing values by interpolation
data = data.interpolate(method='time')

# Save the preprocessed data
data.to_csv('processed_data.csv')
print("Processed data saved to 'processed_data.csv'.")
print(data.head())  # Check the first few rows of the merged data

# Visualize trends
data.plot(subplots=True, figsize=(10, 12), title='Trends of Factors Affecting Home Prices')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Factors')
plt.show()

# Build and evaluate the linear regression model
X = data[['FedFundsRate', 'UnemployedRate', 'CPI', 'HousingStarts']]
y = data['HomePriceIndex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualize feature importance
importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
importance.plot(kind='bar', title='Feature Importance')
plt.show()


