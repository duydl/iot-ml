# IoT Node Setup & Data Collection

This module handles the real-time collection of RSSI (Received Signal Strength Indicator) data from BLE sensor nodes using RIOT OS and the NimBLE stack.

## Hardware Setup

1. **Build and flash TX/RX** (requires two boards):
   - **TX (Transmitter)**: Reads sensor data and notifies the receiver.
   - **RX (Receiver)**: Scans and logs data from multiple TX devices.
   ```bash
   make -C iot/tx flash
   make -C iot/rx flash
   ```

## Data Collection

2. **Log RX CSV** (run from the project root):
   The `log_rx.sh` script collects data and creates a timestamped folder in `iot/data/`.
   ```bash
   ./iot/log_rx.sh
   ```

### CSV Output Format
The data is logged in the following format:
```csv
ts,device,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale,rssi
```
- `ts`: Timestamp from `pyterm`.
- `rssi`: Signal strength in dBm.

### Configuration
- Default port: `/dev/ttyACM0`. Use `PORT=/dev/ttyACM1` to override.
- Default baud: `115200`.

## More Information
- [IOT_COMMUNICATION.md](IOT_COMMUNICATION.md): Detailed explanation of the NimBLE stack, BLE roles, and GATT structure.
