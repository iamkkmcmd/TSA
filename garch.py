import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load time series data into a pandas DataFrame
df = pd.read_csv('time_series_data.csv')

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size
train_data, val_data, test_data = df[0:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]

# Define the hyperparameters to tune
p_values = [1, 2, 3]
o_values = [0, 1]
q_values = [1, 2, 3]

# Define the parameter grid to search over
param_grid = {'p': p_values, 'o': o_values, 'q': q_values}

# Initialize the GARCH model
model = arch_model(train_data['demand'], mean='Zero', vol='GARCH', p=1, o=0, q=1)

# Use TimeSeriesSplit to create training and validation sets
tscv = TimeSeriesSplit(n_splits=5)

# Use GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data['demand'])

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Fit the GARCH model with the best hyperparameters
best_model = arch_model(train_data['demand'], mean='Zero', vol='GARCH', p=best_params['p'], o=best_params['o'], q=best_params['q'])
best_model_fit = best_model.fit()

# Forecast demand using the fitted GARCH model
forecast = best_model_fit.forecast(horizon=test_size)

# Calculate the mean squared error between the forecast and the actual demand
mse = mean_squared_error(test_data['demand'], forecast.mean.iloc[-1, :])

# Plot the actual and forecasted demand
plt.plot(df.index, df['demand'], label='Actual Demand')
plt.plot(df.index[-test_size:], forecast.mean.iloc[-1, :], label='Forecasted Demand')
plt.legend()
plt.show()
