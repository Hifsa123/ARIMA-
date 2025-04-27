# ARIMA-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# Load data
df = pd.read_csv('ETH-USD.csv')

# Show first few rows
df.head()
# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as index and sort
df.set_index('Date', inplace=True)
df = df.sort_index()

# Drop missing values
df = df.dropna()

# Keep only relevant columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df.head()
# Plot Closing Price
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.title('Ethereum (ETH-USD) Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Rolling Mean and Std
rolling_mean = df['Close'].rolling(window=30).mean()
rolling_std = df['Close'].rolling(window=30).std()

plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Original')
plt.plot(rolling_mean, label='30-Day Rolling Mean')
plt.plot(rolling_std, label='30-Day Rolling Std')
plt.legend()
plt.title('Rolling Mean & Std Deviation')
plt.grid()
plt.show()

# ADF Test on original data
result = adfuller(df['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Difference the series if not stationary
df['Close_diff'] = df['Close'].diff()

# Drop NA created by differencing
df.dropna(inplace=True)

# ADF Test on differenced data
result_diff = adfuller(df['Close_diff'])
print('ADF Statistic (Differenced):', result_diff[0])
print('p-value (Differenced):', result_diff[1])
plot_acf(df['Close_diff'], lags=40)
plt.show()

plot_pacf(df['Close_diff'], lags=40)
plt.show()
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['Close'], order=(1, 1, 0))

# Fit the ARIMA model
model_fit = model.fit() # This line is crucial!

# Now you can access the summary using model_fit
print(model_fit.summary())
import matplotlib.pyplot as plt

forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plotting the forecast
plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Historical Data')
plt.plot(forecast_mean, label='Forecast', color='red')

# Shade the forecast area
plt.fill_between(forecast_mean.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='blue', alpha=0.3)

plt.title('Ethereum (ETH-USD) Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
