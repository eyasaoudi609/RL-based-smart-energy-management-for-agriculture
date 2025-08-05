import pandas as pd
input_data = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\Dataset_PV_corrected.csv', delimiter=',' )

# Assuming you have already read your data into input_data
# Convert the 'datetime' column to a datetime index if it's not already
input_data['datetime'] = pd.to_datetime(input_data['datetime'])
input_data.set_index('datetime', inplace=True)

# Resample the data to one-hour intervals and sum the 'power' values within each hour
resampled_data = input_data.resample('1H').sum()

# If you want to fill any missing hours with 0, you can use the fillna method
resampled_data = resampled_data.fillna(0)

# Reset the index to get the datetime as a column again
resampled_data = resampled_data.reset_index()

output_csv_path = 'C:\\Users\\saoudi\\Desktop\\My_Dash\\resampled_PV_data.csv'

# Save the resampled data to a CSV file
resampled_data.to_csv(output_csv_path, index=False)

# Print a message to confirm that the data has been saved
print(f"Resampled data saved to {output_csv_path}")
