import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Load time series data into a pandas DataFrame
df = pd.read_csv('time_series_data.csv')

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size
train_data, val_data, test_data = df[0:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]

# Define the hyperparameters to tune
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

# Use GridSearchCV to find the best hyperparameters
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {'order': [(p, d, q) for p in p_values for d in d_values for q in q_values]}
grid_search = GridSearchCV(ARIMA(train_data), param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data)

# Train the ARIMA model with the best hyperparameters
best_p, best_d, best_q = grid_search.best_params_['order']
arima = ARIMA(train_data, order=(best_p, best_d, best_q))
arima_fit = arima.fit()

# Forecast on the validation and test sets
val_preds = arima_fit.forecast(steps=val_size)[0]
test_preds = arima_fit.forecast(steps=test_size)[0]

# Calculate the mean squared error on the validation and test sets
val_mse = mean_squared_error(val_data, val_preds)
test_mse = mean_squared_error(test_data, test_preds)
print("Validation MSE:", val_mse)
print("Test MSE:", test_mse)

# Plot the actual vs predicted values on the validation and test sets
plt.plot(val_data, label='Actual')
plt.plot(val_preds, label='Predicted')
plt.legend()
plt.show()

plt.plot(test_data, label='Actual')
plt.plot(test_preds, label='Predicted')
plt.legend()
plt.show()
