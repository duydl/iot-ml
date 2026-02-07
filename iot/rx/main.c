/*
 * BLE RX (central): scan, connect, subscribe, and print raw phydat values
 * received from TX as CSV lines.
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "ztimer.h"
#include "host/util/util.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "os/os_mbuf.h"

#define CUSTOM_SVC_UUID     0xff00
#define CUSTOM_CHR_UUID     0xee00
#define DEVICE_NAME_PREFIX  "RIOT-BLE-"
#define DEVICE_NAME_MAX_LEN 31
#ifndef RX_DEBUG
#define RX_DEBUG            1
#endif
#ifndef RX_DEBUG_SCAN
#define RX_DEBUG_SCAN       1
#endif
#ifndef RX_MAX_CONN
#define MAX_CONN            4
#else
#define MAX_CONN            RX_MAX_CONN
#endif

typedef struct __attribute__((packed)) {
    uint16_t seq;
    int16_t temp_val;
    int8_t temp_scale;
    int16_t hum_val;
    int8_t hum_scale;
    int16_t press_val;
    int8_t press_scale;
} sample_t;

static ble_uuid16_t g_svc_uuid = BLE_UUID16_INIT(CUSTOM_SVC_UUID);
static ble_uuid16_t g_chr_uuid = BLE_UUID16_INIT(CUSTOM_CHR_UUID);

static uint8_t g_addr_type;
typedef enum {
    CONN_UNUSED = 0,
    CONN_CONNECTING,
    CONN_CONNECTED,
} conn_state_t;

typedef struct {
    conn_state_t state;
    uint16_t conn_handle;
    uint16_t chr_val_handle;
    uint16_t chr_ccc_handle;
    ble_addr_t addr;
    char name[DEVICE_NAME_MAX_LEN + 1];
} conn_slot_t;

static conn_slot_t g_conns[MAX_CONN];
static uint8_t g_scanning;

static void start_scan(void);

#if RX_DEBUG
#define RX_LOG(...) printf(__VA_ARGS__)
#else
#define RX_LOG(...) do {} while (0)
#endif

#if RX_DEBUG_SCAN
#define RX_SCAN_LOG(...) printf(__VA_ARGS__)
#else
#define RX_SCAN_LOG(...) do {} while (0)
#endif

static void addr_to_str(const ble_addr_t *addr, char *out, size_t out_len)
{
    if (!addr || out_len == 0) {
        return;
    }
    snprintf(out, out_len, "%02x:%02x:%02x:%02x:%02x:%02x",
             addr->val[5], addr->val[4], addr->val[3],
             addr->val[2], addr->val[1], addr->val[0]);
}

static int discover_chr_cb(uint16_t conn_handle, const struct ble_gatt_error *error,
                           const struct ble_gatt_chr *chr, void *arg)
{
    conn_slot_t *slot = (conn_slot_t *)arg;
    (void)error;

    if (chr == NULL) {
        RX_LOG("# RX: chr discovery complete (dev=%s)\n",
               slot ? slot->name : "unknown");
        return 0;
    }

    if (ble_uuid_cmp(&chr->uuid.u, &g_chr_uuid.u) == 0) {
        if (!slot) {
            return 0;
        }
        slot->chr_val_handle = chr->val_handle;
        slot->chr_ccc_handle = chr->val_handle + 1;

        uint16_t ccc_value = 0x0001;
        RX_LOG("# RX: enable notify (ccc=%u, dev=%s)\n",
               slot->chr_ccc_handle, slot->name);
        int rc = ble_gattc_write_flat(conn_handle, slot->chr_ccc_handle,
                                      &ccc_value, sizeof(ccc_value), NULL, NULL);
        if (rc != 0) {
            RX_LOG("# RX: CCC write failed rc=%d\n", rc);
        }
    }

    return 0;
}

static int discover_svc_cb(uint16_t conn_handle, const struct ble_gatt_error *error,
                           const struct ble_gatt_svc *service, void *arg)
{
    conn_slot_t *slot = (conn_slot_t *)arg;
    (void)error;

    if (service == NULL) {
        RX_LOG("# RX: svc discovery complete (dev=%s)\n",
               slot ? slot->name : "unknown");
        return 0;
    }

    RX_LOG("# RX: svc found (start=%u end=%u dev=%s)\n",
           service->start_handle, service->end_handle,
           slot ? slot->name : "unknown");
    ble_gattc_disc_all_chrs(conn_handle,
                            service->start_handle,
                            service->end_handle,
                            discover_chr_cb, slot);
    return 0;
}

static conn_slot_t *find_slot_by_addr(const ble_addr_t *addr)
{
    for (int i = 0; i < MAX_CONN; i++) {
        if (g_conns[i].state != CONN_UNUSED &&
            memcmp(g_conns[i].addr.val, addr->val, sizeof(addr->val)) == 0 &&
            g_conns[i].addr.type == addr->type) {
            return &g_conns[i];
        }
    }
    return NULL;
}

static conn_slot_t *find_slot_by_handle(uint16_t handle)
{
    for (int i = 0; i < MAX_CONN; i++) {
        if (g_conns[i].state != CONN_UNUSED &&
            g_conns[i].conn_handle == handle) {
            return &g_conns[i];
        }
    }
    return NULL;
}

static int active_conn_count(void)
{
    int count = 0;
    for (int i = 0; i < MAX_CONN; i++) {
        if (g_conns[i].state != CONN_UNUSED) {
            count++;
        }
    }
    return count;
}

static int has_connecting(void)
{
    for (int i = 0; i < MAX_CONN; i++) {
        if (g_conns[i].state == CONN_CONNECTING) {
            return 1;
        }
    }
    return 0;
}

static void clear_slot(conn_slot_t *slot)
{
    if (!slot) {
        return;
    }
    memset(slot, 0, sizeof(*slot));
    slot->state = CONN_UNUSED;
}

static conn_slot_t *alloc_slot(const ble_addr_t *addr,
                               const uint8_t *name, uint8_t name_len)
{
    for (int i = 0; i < MAX_CONN; i++) {
        if (g_conns[i].state == CONN_UNUSED) {
            conn_slot_t *slot = &g_conns[i];
            memset(slot, 0, sizeof(*slot));
            slot->state = CONN_CONNECTING;
            slot->addr = *addr;
            uint8_t copy_len = name_len;
            if (copy_len > DEVICE_NAME_MAX_LEN) {
                copy_len = DEVICE_NAME_MAX_LEN;
            }
            memcpy(slot->name, name, copy_len);
            slot->name[copy_len] = '\0';
            return slot;
        }
    }
    return NULL;
}

static int name_matches(const uint8_t *name, uint8_t name_len)
{
    const char *prefix = DEVICE_NAME_PREFIX;
    size_t prefix_len = strlen(prefix);

    if (!name || name_len <= prefix_len) {
        return 0;
    }
    if (memcmp(name, prefix, prefix_len) != 0) {
        return 0;
    }

    int saw_digit = 0;
    for (size_t i = prefix_len; i < name_len; i++) {
        char c = (char)name[i];
        if (c >= '0' && c <= '9') {
            saw_digit = 1;
            continue;
        }
        if (c == '/' && saw_digit) {
            if (i + 1 >= name_len) {
                return 0;
            }
            for (i = i + 1; i < name_len; i++) {
                c = (char)name[i];
                if (c < '0' || c > '9') {
                    return 0;
                }
            }
            return 1;
        }
        return 0;
    }
    return saw_digit;
}

static int gap_event(struct ble_gap_event *event, void *arg)
{
    conn_slot_t *slot = (conn_slot_t *)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT: {
        char addr_str[18] = {0};
        if (slot) {
            addr_to_str(&slot->addr, addr_str, sizeof(addr_str));
        } else {
            strncpy(addr_str, "<unknown>", sizeof(addr_str));
            addr_str[sizeof(addr_str) - 1] = '\0';
        }
        if (event->connect.status != 0) {
            RX_LOG("# RX: connect failed status=%d addr=%s\n",
                   event->connect.status, addr_str);
            if (slot) {
                clear_slot(slot);
            }
            start_scan();
            return 0;
        }
        if (slot) {
            slot->state = CONN_CONNECTED;
            slot->conn_handle = event->connect.conn_handle;
            RX_LOG("# RX: connected handle=%u dev=%s addr=%s\n",
                   slot->conn_handle, slot->name, addr_str);
        }

        int rc = ble_gattc_disc_svc_by_uuid(event->connect.conn_handle,
                                            &g_svc_uuid.u, discover_svc_cb,
                                            slot);
        if (rc != 0) {
            RX_LOG("# RX: service discovery failed rc=%d\n", rc);
            ble_gap_terminate(event->connect.conn_handle,
                              BLE_ERR_REM_USER_CONN_TERM);
        }
        start_scan();
        return 0;
    }

    case BLE_GAP_EVENT_DISCONNECT:
        RX_LOG("# RX: disconnected reason=%d\n", event->disconnect.reason);
        if (slot) {
            clear_slot(slot);
        } else {
            clear_slot(find_slot_by_handle(event->disconnect.conn.conn_handle));
        }
        start_scan();
        return 0;

    case BLE_GAP_EVENT_NOTIFY_RX: {
        if (event->notify_rx.om->om_len < sizeof(sample_t)) {
            RX_LOG("# RX: short notify len=%u\n",
                   (unsigned)event->notify_rx.om->om_len);
            return 0;
        }

        sample_t sample;
        os_mbuf_copydata(event->notify_rx.om, 0, sizeof(sample), &sample);

        if (!slot) {
            slot = find_slot_by_handle(event->notify_rx.conn_handle);
        }
        const char *dev_name = slot ? slot->name : "unknown";

        int8_t rssi = 127;  // 127 often used as "unknown/unavailable"
        ble_gap_conn_rssi(event->notify_rx.conn_handle, &rssi); 
        printf("%s,%u,%d,%d,%d,%d,%d,%d,%d\n",
               dev_name,
               sample.seq,
               sample.temp_val, sample.temp_scale,
               sample.hum_val, sample.hum_scale,
               sample.press_val, sample.press_scale, rssi);

        return 0;
    }
    }

    return 0;
}

static int scan_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    struct ble_hs_adv_fields fields;
    memset(&fields, 0, sizeof(fields));

    switch (event->type) {
    case BLE_GAP_EVENT_DISC_COMPLETE:
        g_scanning = 0;
        RX_LOG("# RX: scan complete\n");
        start_scan();
        return 0;

    case BLE_GAP_EVENT_DISC:
        {
            int rc = ble_hs_adv_parse_fields(&fields, event->disc.data,
                                             event->disc.length_data);
            if (rc != 0) {
                RX_SCAN_LOG("# RX: adv parse failed rc=%d\n", rc);
                return 0;
            }
        }

        int uuid_match = 0;
        if (fields.uuids16 != NULL && fields.num_uuids16 > 0) {
            for (int i = 0; i < fields.num_uuids16; i++) {
                if (ble_uuid_cmp(&g_svc_uuid.u, &fields.uuids16[i].u) == 0) {
                    uuid_match = 1;
                    break;
                }
            }
        }

        int name_match = 0;
        char name_buf[DEVICE_NAME_MAX_LEN + 1];
        name_buf[0] = '\0';
        if (fields.name != NULL && fields.name_len > 0) {
            if (name_matches(fields.name, fields.name_len)) {
                name_match = 1;
            }
            uint8_t copy_len = fields.name_len;
            if (copy_len > DEVICE_NAME_MAX_LEN) {
                copy_len = DEVICE_NAME_MAX_LEN;
            }
            memcpy(name_buf, fields.name, copy_len);
            name_buf[copy_len] = '\0';
        } else {
            strncpy(name_buf, "<none>", sizeof(name_buf));
            name_buf[sizeof(name_buf) - 1] = '\0';
        }

        if (uuid_match || (fields.name && fields.name_len > 0)) {
            char addr_str[18] = {0};
            addr_to_str(&event->disc.addr, addr_str, sizeof(addr_str));
            RX_SCAN_LOG("# RX: adv addr=%s rssi=%d name=%s uuid=%d name_match=%d\n",
                        addr_str, event->disc.rssi, name_buf,
                        uuid_match, name_match);
        }

        if (uuid_match && name_match) {
            if (active_conn_count() >= MAX_CONN) {
                RX_LOG("# RX: skip %s (max conn reached)\n", name_buf);
                return 0;
            }
            if (find_slot_by_addr(&event->disc.addr)) {
                RX_LOG("# RX: skip %s (already tracked)\n", name_buf);
                return 0;
            }
            conn_slot_t *slot = alloc_slot(&event->disc.addr,
                                           fields.name,
                                           fields.name_len);
            if (!slot) {
                RX_LOG("# RX: no free slot for %s\n", name_buf);
                return 0;
            }
            RX_LOG("# RX: found %s, connecting...\n", slot->name);
            int cancel_rc = ble_gap_disc_cancel();
            if (cancel_rc != 0) {
                RX_LOG("# RX: scan cancel failed rc=%d\n", cancel_rc);
            }
            g_scanning = 0;
            int rc = ble_gap_connect(g_addr_type, &(event->disc.addr), 100,
                                     NULL, gap_event, slot);
            if (rc != 0) {
                RX_LOG("# RX: connect start failed rc=%d\n", rc);
                clear_slot(slot);
                start_scan();
            }
        }
        return 0;
    }

    return 0;
}

static void start_scan(void)
{
    if (g_scanning) {
        RX_LOG("# RX: scan already active\n");
        return;
    }
    if (has_connecting()) {
        RX_LOG("# RX: scan blocked (connecting)\n");
        return;
    }
    if (active_conn_count() >= MAX_CONN) {
        RX_LOG("# RX: scan blocked (max conn=%d)\n", MAX_CONN);
        return;
    }
    const struct ble_gap_disc_params scan_params = { 10000, 200, 0, 0, 0, 1 };
    int rc = ble_gap_disc(g_addr_type, 100, &scan_params, scan_event, NULL);
    if (rc != 0) {
        RX_LOG("# RX: scan failed rc=%d\n", rc);
        return;
    }
    g_scanning = 1;
    RX_LOG("# RX: scan started (max_conn=%d)\n", MAX_CONN);
}

int main(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    rc = ble_hs_id_infer_auto(0, &g_addr_type);
    assert(rc == 0);

    printf("device,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale\n");

    start_scan();

    while (1) {
        ztimer_sleep(ZTIMER_MSEC, 1000);
    }

    return 0;
}
