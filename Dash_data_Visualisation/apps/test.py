import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# Load the trained XGBoost model
model = joblib.load('power_prediction_model.pkl')  # Replace with your model file path

# Load input data from a CSV file with the correct delimiter and encoding
input_data = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\historical_data.csv', delimiter=',', encoding='latin1')
# Create a time series of future datetime values for the next 24 hours
end_datetime = datetime.datetime.now()  # Replace with your desired start datetime
forecast_period_hours = 24
forecast_datetime = end_datetime + pd.to_timedelta(np.arange(1, forecast_period_hours + 1), unit='H')

# Create an empty DataFrame to store the predicted power values
forecast_data = pd.DataFrame({'datetime': forecast_datetime, 'power': None})

# Predict the power for each future time point
for i, forecast_time in enumerate(forecast_datetime):
    forecast_input_data = input_data.copy()
    forecast_input_data['hour'] = forecast_time.hour
    forecast_input_data['minute'] = forecast_time.minute
    forecast_power = model.predict(forecast_input_data)
    forecast_data.at[i, 'power'] = forecast_power[0]  # Store the predicted power value

# Plot the predicted power values
plt.figure(figsize=(12, 6))
plt.plot(forecast_data['datetime'], forecast_data['power'], marker='o', linestyle='-')
plt.title('Predicted Power for the Next 24 Hours')
plt.xlabel('Datetime')
plt.ylabel('Power')
plt.grid(True)

# Customize the x-axis ticks (display hours at 2-hour intervals)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))  # Set tick positions at 2-hour intervals

# Format the x-axis tick labels to show only hours
def format_hour(x, pos):
    return forecast_datetime[pos].strftime('%H:%M')

ax.xaxis.set_major_formatter(FuncFormatter(format_hour))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()