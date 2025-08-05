import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 100)
SHIFTABLE = ['cleaning_systemC', 'lightning_systemC', 'osmose','water_pump','lightning_system_R']
TIME_SHIFTABLE = ['milk_pumpC', 'vacuum_pump1', 'vacuum_pump2', 'cooling_systemC', 'RO1','RO2','cooling_system_R','vacuum_pumpC']
NON_SHIFTABLE = ['coldroomF']
DEVICES = ['cleaning_system', 'cooling_system', 'lightning_system', 'milk_pump', 'RO','vacuum_pump' , 'osmose', 'coldroom','water_pump','non-shiftable']
INCLUDE = ['dataid', 'cleaning_system', 'cooling_system', 'lightning_system', 'milk_pump', 'RO','vacuum_pump' , 'osmose', 'coldroom','water_pump', 'non-shiftable', 'total']#timeshiftable aka include

df = pd.read_csv('C:\\Users\\saoudi\\Desktop\\final\\Data\\10min_all_data.csv', delimiter=',', engine='python', encoding="ISO-8859-1", parse_dates=['local_10min'], dayfirst=True, index_col=['local_10min'])
df.index = pd.to_datetime(df.index, utc=True, infer_datetime_format=True)
df.index.names = ['time']
df = df.tz_convert(None)
df = df.groupby(['dataid']).resample('10T').max()
df = df.drop('dataid', axis=1).reset_index('dataid')
df = df.fillna(0)
df = df.apply(lambda l: np.where(l < 0.1, 0, l))

df['RO'] = df[['RO1', 'RO2']].sum(axis=1)#.clip(upper=4.0)  
df['vacuum_pump'] = df[['vacuum_pump1', 'vacuum_pump2','vacuum_pumpC']].sum(axis=1) 
df['lightning_system'] = df[['lightning_system', 'lightning_system_R']].sum(axis=1)
df['cooling_system'] = df[['cooling_systemC', 'cooling_system_R']].sum(axis=1)
df['cleaning_system'] = df['cleaning_systemC']
df['milk_pump'] = df['milk_pumpC']
df['osmose'] = df['osmoseF']
df['water_pump'] = df['water_pumpF']
df['coldroom'] = df['coldroomF']

for device, consumption, threshold in zip(TIME_SHIFTABLE, [4, 1, 2, 2], [0.1, 0.1, 0.1, 0.1]):
    df[device] = df[device].apply(lambda x: consumption if x >= threshold else 0)

df['non-shiftable'] = df[NON_SHIFTABLE].sum(axis=1).clip(upper=5.0)
df['total'] = df[DEVICES].sum(axis=1)

# Uncomment to save processed data to csv
df[INCLUDE].to_csv('C:\\Users\\saoudi\\Desktop\\final\\Data\\10min_all_data_new.csv')
# Filter Household
dataid = 2
# df = df.loc[df['dataid'] != 9019]
df = df.loc[df['dataid'] == dataid]

# Filter dates
day = 152
start_date = datetime.datetime.strptime('{} {}'.format(day, 2022), '%j %Y')
end_date = datetime.datetime.strptime('{} {}'.format(day + 1, 2022), '%j %Y')
# start_date = datetime.datetime(2018, 7, 9)
# end_date = datetime.datetime(2018, 7, 10)
df = df.loc[(df.index >= start_date) & (df.index < end_date)]

# create the plot
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})
solar_plot = df[DEVICES].plot(linewidth=0.5, marker='.')
solar_plot.set_xlabel('Date')
solar_plot.set_ylabel('Grid Usage kW')

# display the plot
plt.title('Major consumers')
plt.ylabel('Power consumnption (KW)')
plt.show()