import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress

AGENT_IDS = [1, 2, 3]
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Specify the column names in your CSV file
column_names = ['time', 'power']


# Load PV power data (replace this with your data loading logic)
df_pv_power = pd.read_csv('C:\\Users\\saoudi\\Desktop\\final\\Data\\pv_data_10min.csv',delimiter=',',
                          parse_dates=['time'],
                          index_col=['time'],
                             usecols=column_names)

# Check for duplicate timestamps
duplicate_timestamps = df_pv_power.index.duplicated()
if any(duplicate_timestamps):
    print("Duplicate timestamps found. Removing duplicates...")
    df_pv_power = df_pv_power[~duplicate_timestamps]

# Load appliances power data (replace this with your data loading logic)
df_appliances_power = pd.read_csv('C:\\Users\\saoudi\\Desktop\\final\\Data\\10min_all_data_new.csv',delimiter=',',
                                  parse_dates=['time'],
                                  index_col=['time'])





start, end, steps = 151, 165, 144
baselines = np.zeros((len(AGENT_IDS), len(range(start, end)), steps))

for id, agent in enumerate(AGENT_IDS):
    df_filter = df_appliances_power.loc[df_appliances_power['dataid'] == agent]
    
    # Resample the PV power data using asfreq to match the time steps of the appliances data
    df_pv_power_resampled = df_pv_power.asfreq('10T', method='pad')




start_doy, end_doy, steps = 151, 165, 144
baselines = np.zeros((len(AGENT_IDS), end_doy - start_doy + 1, steps))

for id, agent in enumerate(AGENT_IDS):
    df_filter = df_appliances_power.loc[df_appliances_power['dataid'] == agent]
    
    for day_doy in range(start_doy, end_doy + 1):
        for step in range(steps):
            # Calculate the timestamp for the current step
            time_delta = datetime.timedelta(minutes=step * 10)
            current_date = datetime.datetime(2022, 1, 1) + datetime.timedelta(days=day_doy - 1)
            current_time = current_date + time_delta
            
            try:
                # Get PV power and appliances power for the current time step
                pv_power = df_pv_power_resampled.loc[current_time]['power']
                appliances_power = df_filter.loc[current_time]['cleaning_system'] + df_filter.loc[current_time]['milk_pump'] + df_filter.loc[current_time]['osmose'] + df_filter.loc[current_time]['coldroom'] + df_filter.loc[current_time]['water_pump'] + df_filter.loc[current_time]['cooling_system'] + df_filter.loc[current_time]['RO'] + df_filter.loc[current_time]['vacuum_pump'] + df_filter.loc[current_time]['lightning_system']
                # Calculate the total power demand
                total_power_demand = appliances_power

                # Decide whether to use PV power or grid power
                if pv_power >= total_power_demand:
                    baseline = total_power_demand  # Use PV power
                else:
                    baseline = total_power_demand  # Use grid power

                baselines[id][day_doy - start_doy][step] = baseline

                print(agent, day_doy, step, baseline)
            except KeyError:
                print(f"Data not available for {current_time}")



# Save the baselines array (adjust the path accordingly)
np.save('C:\\Users\\saoudi\\Desktop\\final\\Data\\baselines.npy', baselines)
