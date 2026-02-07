/*
 * Simple BLE RX (Central) that scans for the TX device, connects,
 * subscribes to notifications, and prints received data.
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

#define CUSTOM_SVC_UUID     0xff00
#define CUSTOM_CHR_UUID     0xee00
#define TARGET_NAME         "RIOT-IOT-TX"

static ble_uuid16_t g_svc_uuid = BLE_UUID16_INIT(CUSTOM_SVC_UUID);
static ble_uuid16_t g_chr_uuid = BLE_UUID16_INIT(CUSTOM_CHR_UUID);

static uint8_t g_addr_type;
static uint8_t g_conn_state;
static uint8_t g_notify_state;
static uint16_t g_conn_handle;
static uint16_t g_chr_val_handle;
static uint16_t g_chr_ccc_handle;

static void start_scan(void);

static int discover_chr_cb(uint16_t conn_handle, const struct ble_gatt_error *error,
                           const struct ble_gatt_chr *chr, void *arg)
{
    (void)arg;
    (void)error;

    if (chr == NULL) {
        return 0;
    }

    if (ble_uuid_cmp(&chr->uuid.u, &g_chr_uuid.u) == 0) {
        g_chr_val_handle = chr->val_handle;
        g_chr_ccc_handle = chr->val_handle + 1; /* CCCD usually follows value */

        uint16_t ccc_value = 0x0001;
        printf("RX: enabling notify (ccc handle=%u)\n", g_chr_ccc_handle);
        int rc = ble_gattc_write_flat(conn_handle, g_chr_ccc_handle,
                                      &ccc_value, sizeof(ccc_value), NULL, NULL);
        if (rc != 0) {
            printf("RX: CCC write failed rc=%d\n", rc);
        }
    }

    return 0;
}

static int discover_svc_cb(uint16_t conn_handle, const struct ble_gatt_error *error,
                           const struct ble_gatt_svc *service, void *arg)
{
    (void)arg;
    (void)error;

    if (service == NULL) {
        return 0;
    }

    printf("RX: service UUID=0x%04x\n", ble_uuid_u16((ble_uuid_t *)&service->uuid));
    ble_gattc_disc_all_chrs(conn_handle,
                            service->start_handle,
                            service->end_handle,
                            discover_chr_cb, NULL);
    return 0;
}

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT: {
        if (event->connect.status != 0) {
            printf("RX: connect failed status=%d\n", event->connect.status);
            g_conn_state = 0;
            g_notify_state = 0;
            start_scan();
            return 0;
        }
        g_conn_state = 1;
        g_notify_state = 0;
        g_conn_handle = event->connect.conn_handle;

        int rc = ble_gattc_disc_svc_by_uuid(g_conn_handle, &g_svc_uuid.u,
                                            discover_svc_cb, NULL);
        if (rc != 0) {
            printf("RX: service discovery failed\n");
            ble_gap_terminate(g_conn_handle, BLE_ERR_REM_USER_CONN_TERM);
        }
        return 0;
    }

    case BLE_GAP_EVENT_DISCONNECT:
        printf("RX: disconnected reason=%d\n", event->disconnect.reason);
        g_conn_state = 0;
        g_notify_state = 0;
        start_scan();
        return 0;

    case BLE_GAP_EVENT_NOTIFY_RX:
        printf("RX: data -> ");
        for (uint8_t i = 0; i < event->notify_rx.om->om_len; i++) {
            printf("%02x ", event->notify_rx.om->om_data[i]);
        }
        printf("\n");
        g_notify_state = 1;
        return 0;
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
        start_scan();
        return 0;

    case BLE_GAP_EVENT_DISC:
        ble_hs_adv_parse_fields(&fields, event->disc.data, event->disc.length_data);

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
        if (fields.name != NULL && fields.name_len > 0) {
            if (strlen(TARGET_NAME) == fields.name_len &&
                memcmp(fields.name, TARGET_NAME, fields.name_len) == 0) {
                name_match = 1;
            }
        }

        if (uuid_match && name_match) {
            printf("RX: found %.*s, connecting...\n",
                   fields.name_len, fields.name);
            ble_gap_disc_cancel();
            ble_gap_connect(g_addr_type, &(event->disc.addr), 100,
                            NULL, gap_event, NULL);
        }
        return 0;
    }

    return 0;
}

static void start_scan(void)
{
    /* itvl, window, filter_policy, limited, passive, filter_duplicates */
    const struct ble_gap_disc_params scan_params = { 10000, 200, 0, 0, 0, 1 };
    int rc = ble_gap_disc(g_addr_type, 100, &scan_params, scan_event, NULL);
    if (rc != 0) {
        printf("RX: scan failed rc=%d\n", rc);
    }
}

int main(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    rc = ble_hs_id_infer_auto(0, &g_addr_type);
    assert(rc == 0);

    start_scan();

    int tick = 0;
    while (1) {
        if (tick % 3 == 0) {
            if (!g_conn_state) {
                printf("RX: not connected\n");
            } else if (!g_notify_state) {
                printf("RX: connected, waiting for notify\n");
            } else {
                printf("RX: connected, receiving\n");
            }
        }
        tick++;
        ztimer_sleep(ZTIMER_SEC, 1);
    }

    return 0;
}
