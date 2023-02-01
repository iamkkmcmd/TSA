import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load time series data into a pandas DataFrame
df = pd.read_csv('time_series_data.csv')

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size
train_data, val_data, test_data = df[0:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]

# Define the hyperparameters to tune
trend_values = ['add', 'additive', 'multiplicative']
damped_values = [True, False]
seasonal_values = ['add', 'additive', 'multiplicative']
seasonal_periods = [12, 24, 52]

# Define the parameter grid to search over
param_grid = {'trend': trend_values, 'damped': damped_values, 'seasonal': seasonal_values, 'seasonal_periods': seasonal_periods}

# Initialize the Holt-Winters model
model = ExponentialSmoothing(train_data['demand'], trend='add', seasonal='multiplicative', seasonal_periods=12)

# Use TimeSeriesSplit to create training and validation sets
tscv = TimeSeriesSplit(n_splits=5)

# Use GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data['demand'])

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Fit the Holt-Winters model with the best hyperparameters
best_model = ExponentialSmoothing(train_data['demand'], trend=best_params['trend'], damped=best_params['damped'], seasonal=best_params['seasonal'], seasonal_periods=best_params['seasonal_periods'])
best_model_fit = best_model.fit()

# Forecast demand using the fitted Holt-Winters model
forecast = best_model_fit.forecast(steps=test_size)

# Calculate the mean squared error between the forecast and the actual demand
mse = mean_squared_error(test_data['demand'], forecast)

# Plot the actual and forecasted demand
plt.plot(df.index, df['demand'], label='Actual Demand')
plt.plot(df.index[-test_size:], forecast, label='Forecasted Demand')
plt.xlabel('Time')
plt.ylabel('Demand')
