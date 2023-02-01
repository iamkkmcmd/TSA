import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load time series data into a pandas DataFrame
df = pd.read_csv('time_series_data.csv')

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size
train_data, val_data, test_data = df[0:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]

# Define the hyperparameters to tune
smoothing_level = [0.1, 0.3, 0.5, 0.7, 0.9]
smoothing_slope = [0.01, 0.03, 0.05, 0.07, 0.09]
smoothing_seasonal = [0.1, 0.3, 0.5, 0.7, 0.9]

# Define the parameter grid to search over
param_grid = {'smoothing_level': smoothing_level,
              'smoothing_slope': smoothing_slope,
              'smoothing_seasonal': smoothing_seasonal}

# Use TimeSeriesSplit to create training and validation sets
tscv = TimeSeriesSplit(n_splits=5)

# Initialize the exponential smoothing model
model = sm.tsa.ExponentialSmoothing

# Use GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data['demand'])

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Fit the exponential smoothing model with the best hyperparameters
best_model = sm.tsa.ExponentialSmoothing(train_data['demand'], trend=None, seasonal=None, seasonal_periods=12)
best_model_fit = best_model.fit(smoothing_level=best_params['smoothing_level'], smoothing_slope=best_params['smoothing_slope'], smoothing_seasonal=best_params['smoothing_seasonal'])

# Forecast demand using the fitted exponential smoothing model
forecast = best_model_fit.forecast(steps=test_size)

# Calculate the mean squared error between the forecast and the actual demand
mse = mean_squared_error(test_data['demand'], forecast)

# Plot the actual and forecasted demand
plt.plot(df.index, df['demand'], label='Actual Demand')
plt.plot(df.index[-test_size:], forecast
