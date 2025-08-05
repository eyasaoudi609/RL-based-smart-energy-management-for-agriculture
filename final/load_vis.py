import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\saoudi\\Desktop\\Eya-PFE\\Data_RL\\Data_farms\\Farm1\\box_osmose.csv', delimiter=',')
# Calculate the time values in hours
num_samples = len(df['watt'])
time_in_hours = np.arange(num_samples) / (num_samples / 24)

fig = plt.figure(figsize=(18, 12))
plt.plot(time_in_hours[0:86400 ], df['watt'][0:86400 ])  # Adjust 'time_in_hours' and 'power' as needed
plt.title("PV Power Production", fontsize=22)
plt.xlabel('Time (Hour)', fontsize=22)
plt.ylabel('Power (W)', fontsize=22)
plt.grid(color='green', linestyle='--', linewidth=0.5)
# Uncomment the following line if you want to add a legend
# plt.legend()

plt.show()
