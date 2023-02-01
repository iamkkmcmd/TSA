import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Load the time series data into a pandas DataFrame
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
P_values = [0, 1, 2]
D_values = [0, 1]
Q_values = [0, 1, 2]
m_values = [1, 2, 3, 4, 12]

# Define the parameter grid to search over
param_grid = {'order': [(p, d, q), (p, d, q, P, D, Q, m) for p in p_values for d in d_values for q in q_values for P in P_values for D in D_values for Q in Q_values for m in m_values]}

# Initialize the SARIMAX model
model = SARIMAX(train_data['demand'], order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))

# Use TimeSeriesSplit to create training and validation sets
tscv = TimeSeriesSplit(n_splits=5)

# Use GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data['demand'])

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Fit the SARIMAX model with the best hyperparameters
best_model = SARIMAX(train_data['demand'], order=best_params['order'][0], seasonal_order=best_params['order'][1])
best_model_fit = best_model.fit()

# Forecast demand using the fitted SARIMAX model
forecast = best_model_fit.forecast(steps=test_size)

# Calculate the mean squared error between the forecast and the actual demand
mse = mean_squared_error(test_data['demand'], forecast)

# Plot the actual and forecasted demand
plt.plot(df.index, df['demand'], label='Actual Demand')
plt.plot(df.index[-test_size:])