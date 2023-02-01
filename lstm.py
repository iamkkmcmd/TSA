import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
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
batch_size = [16, 32, 64, 128]
epochs = [10, 20, 30]
optimizer = ['SGD', 'RMSprop', 'Adam']

# Define the parameter grid to search over
param_grid = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer}

# Define a function to create the LSTM model
def create_model(batch_size=16, epochs=10, optimizer='Adam'):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Use KerasRegressor to wrap the LSTM model
model = KerasRegressor(build_fn=create_model)

# Use TimeSeriesSplit to create training and validation sets
tscv = TimeSeriesSplit(n_splits=5)

# Use GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(train_data, train_data['demand'])

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Fit the LSTM model with the best hyperparameters
best_model = create_model(batch_size=best_params['batch_size'], epochs=best_params['epochs'], optimizer=best_params['optimizer'])
best_model.fit(train_data, train_data['demand'], batch_size=best_params['batch_size'], epochs=best_params['epochs'])

# Forecast demand using the fitted LSTM model
forecast = best_model.predict(test_data)
