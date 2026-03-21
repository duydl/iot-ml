#Plots Timedelay of each packet compared to when it should arrive, if the 10Hz rythm were perfectly kept after the first considered datapoint
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
    sensor_data = df[df['device'] == sensor].head(4380).tail(4000) #ignores the first 380 and then plots 4000
    sensor_data['ts'] = pd.to_datetime(sensor_data['ts'])    


    #Comparison to rigid grid
    starttime=sensor_data['ts'].iloc[0]
    end_time = sensor_data['ts'].iloc[-1] +pd.DateOffset(seconds=5) #5s buffer
    print(starttime)
    gridValues=pd.date_range(start=starttime, end=end_time, freq='100ms')[1:]
    
    deltas = []
    for i in range(len(sensor_data)):
        delta = gridValues[i] - sensor_data['ts'].iloc[i]
        deltas.append(delta)

    # convert timedelta to seconds
    timestamps = [x.total_seconds() for x in deltas]

    # Plot the time differences
    ax.scatter(range(len(timestamps)),timestamps, s=10, c=color)
    ax.set_title(f'Sensor {sensor}')


plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


