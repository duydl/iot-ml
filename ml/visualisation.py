#visualisation of raw data in one file
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.ticker import FuncFormatter

#custom formatter for plotticks
def format_milliseconds(x, pos=None):
    dt=dates.num2date(x)
    return f"{dt.second:02d}.{dt.microsecond//1000:03d}"


# Load the RSSI data
df = pd.read_csv('../data/20260307_173911_garden/rx.csv', sep=',', header=0)

sensors = ['RIOT-BLE-0', 'RIOT-BLE-1', 'RIOT-BLE-2', 'RIOT-BLE-3']
colors = ['blue', 'green', 'red', 'purple']

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for sensor, color, ax in zip(sensors, colors, axs.flatten()):
    sensor_data = df[df['device'] == sensor]#.head(380).tail(30)
    sensor_data['ts'] = pd.to_datetime(sensor_data['ts'])    

    #plotting data
    ax.scatter(sensor_data['ts'], sensor_data['rssi'], s=10, c=color)
    ax.set_title(f'Sensor {sensor}')
    #The following 4 lines draw a 10Hz grid for comparison (only possible on small enough segments)
    #ax.xaxis.set_major_locator(dates.MicrosecondLocator(10**5))        
    #ax.xaxis.set_major_formatter(FuncFormatter(format_milliseconds))
    #ax.tick_params(axis ='x', rotation=70)
    #ax.grid(True)
    ax.set_ylim(-85,-50)


plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


