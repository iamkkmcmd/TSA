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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# Load dataset
df = pd.read_csv("data.csv")

# Plot dataset
sns.lineplot(data=df)
plt.show()

# Split dataset into training and test sets
split = int(len(df) * 0.7)
train = df.iloc[:split, :]
test = df.iloc[split:, :]

# Initialize Holt Winters Model
model = ExponentialSmoothing(train["value"], trend="add", seasonal="multiplicative", seasonal_periods=12)

# Hyperparameter tuning
best_params = None
best_aic = float("inf")
for alpha in np.arange(0, 1, 0.1):
    for beta in np.arange(0, 1, 0.1):
        for gamma in np.arange(0, 1, 0.1):
            try:
                model = ExponentialSmoothing(train["value"], trend="add", seasonal="multiplicative", seasonal_periods=12, 
                                            damped=False, smoothing_level=alpha, smoothing_slope=beta, 
                                            smoothing_seasonal=gamma)
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_params = (alpha, beta, gamma)
            except:
                pass

# Fit model with best hyperparameters
model = ExponentialSmoothing(train["value"], trend="add", seasonal="multiplicative", seasonal_periods=12, 
                            damped=False, smoothing_level=best_params[0], smoothing_slope=best_params[1], 
                            smoothing_seasonal=best_params[2])
fit = model.fit()

# Plot forecast
forecast = fit.forecast(len(test))
plt.plot(train["value"], label="Train")
plt.plot(test["value"], label="Test")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.show()
