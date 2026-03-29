# IoT Communication: RX & TX

This code implements a BLE (Bluetooth Low Energy) sensor network using RIOT OS and the NimBLE stack. It consists of two parts:
1.  **TX (Transmitter/Peripheral)**: Reads sensor data and notifies a central device.
2.  **RX (Receiver/Central)**: Scans, connects, and collects data from multiple TX devices.

## 1. Concept Theory

### Bluetooth Low Energy (BLE) Roles
*   **Gap (Generic Access Profile)**: Defines roles for device discovery and connection.
    *   **Peripheral (TX)**: Advertises its presence to let others know it exists. It is the "server" that holds data.
    *   **Central (RX)**: Scans for advertisements and initiates connections. It is the "client" that requests data.
*   **GATT (Generic Attribute Profile)**: Defines how data is structured and transferred.
    *   **Service**: A collection of data and associated behaviors (e.g., specific sensor service).
    *   **Characteristic**: A specific piece of data within a service (e.g., temperature value).
    *   **Notification**: A mechanism where the server pushes data to the client whenever it changes (or periodically), without the client polling.

### Data Flow
1.  **Advertising**: TX broadcasts packets containing its name and Service UUID (`0xff00`).
2.  **Scanning**: RX listens for these packets.
3.  **Connection**: RX initiates a connection to a specific TX address.
4.  **Discovery**: RX asks TX what services and characteristics it has.
5.  **Subscription**: RX writes to the **CCC (Client Characteristic Configuration)** descriptor to enable notifications.
6.  **Transmission**: TX sends sensor data as notifications. RX receives them and processes the data (prints to CSV).

## 2. Library & Syntax (RIOT + NimBLE)

The project uses the **NimBLE** stack (a lightweight BLE stack for Apache Mynewt, ported to RIOT).

### Key Libraries
*   `host/ble_gap.h`: Functions for advertising, scanning, and connecting.
*   `host/ble_gatt.h`: Functions for defining services/characteristics and handling read/write/notify operations.
*   `saul_reg.h`: (TX only) **S**ensor **A**ctuator **U**ber **L**ayer. Used to read hardware sensors in a hardware-agnostic way.
*   `ztimer.h`: System timer for periodic events (sleeping loops).

### TX Syntax (Peripheral)

1.  **Defining Services & Characteristics**:
    Uses a struct array to define the GATT table.
    ```c
    static const struct ble_gatt_svc_def gatt_svcs[] = {
        {
            .type = BLE_GATT_SVC_TYPE_PRIMARY,
            .uuid = BLE_UUID16_DECLARE(0xff00), // Service UUID
            .characteristics = (struct ble_gatt_chr_def[]) {
                {
                    .uuid = BLE_UUID16_DECLARE(0xee00), // Characteristic UUID
                    .access_cb = gatt_access_cb,        // Callback for read/write
                    .val_handle = &g_notify_val_handle, // Handle to reference this char later
                    .flags = BLE_GATT_CHR_F_NOTIFY,     // Supports notifications
                },
                { 0 },
            },
        },
        { 0 },
    };
    ```

2.  **Advertising**:
    Configured via `ble_gap_adv_set_fields` and started with `ble_gap_adv_start`.
    *   **Syntax**: `fields.name` sets the device name seen by scanners. `fields.uuids16` includes the service UUID in the advertisement packet so RX knows it's the right device.

3.  **Sending Data (Notification)**:
    *   Data is packed into an `os_mbuf` (memory buffer).
    *   Sent using `ble_gatts_notify_custom`.
    ```c
    // Allocate buffer with data
    struct os_mbuf *om = ble_hs_mbuf_from_flat(&sample, sizeof(sample));
    // Send notification
    ble_gatts_notify_custom(g_conn_handle, g_notify_val_handle, om);
    ```

### RX Syntax (Central)

1.  **Scanning**:
    *   Uses `ble_gap_disc` to start scanning.
    *   **Callback**: `scan_event` handles `BLE_GAP_EVENT_DISC` (advertisement received).
    *   **Logic**: Checks if `fields.uuids16` matches our target UUID or `fields.name` matches the prefix.

2.  **Connecting**:
    *   `ble_gap_connect` initiates connection.
    *   **Callback**: `gap_event` handles `BLE_GAP_EVENT_CONNECT`.

3.  **Service Discovery**:
    *   `ble_gattc_disc_svc_by_uuid`: Finds the custom service on the remote TX.
    *   `ble_gattc_disc_all_chrs`: Finds characteristics within that service.

4.  **Subscribing**:
    *   Writes `0x0001` to the CCC descriptor (handle + 1 usually, or discovered descriptor).
    *   `ble_gattc_write_flat(conn_handle, ccc_handle, &value, ...)`

5.  **Receiving Data**:
    *   Handled in `gap_event` under `BLE_GAP_EVENT_NOTIFY_RX`.
    *   Data extracted using `os_mbuf_copydata`.

## 3. Communication Protocol (Application Layer)

The application defines a custom binary struct (`sample_t`) for efficient transfer.

```c
typedef struct __attribute__((packed)) {
    uint16_t seq;       // Sequence number
    int16_t temp_val;   // Temperature value
    int8_t temp_scale;  // Temperature scale factor
    int16_t hum_val;    // Humidity value
    ...
} sample_t;
```

*   **`__attribute__((packed))`**: Ensures no padding bytes are added by the compiler, so the binary size is consistent across devices.
*   **PHYDAT**: The sensor data format uses RIOT's `phydat_t` logic (value + scale), which is preserved in the network packet.

