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
#define TARGET_NAME         "RIOT-IOT-TX"

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
        g_chr_ccc_handle = chr->val_handle + 1;

        uint16_t ccc_value = 0x0001;
        printf("# RX: enable notify (ccc=%u)\n", g_chr_ccc_handle);
        int rc = ble_gattc_write_flat(conn_handle, g_chr_ccc_handle,
                                      &ccc_value, sizeof(ccc_value), NULL, NULL);
        if (rc != 0) {
            printf("# RX: CCC write failed rc=%d\n", rc);
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
            printf("# RX: connect failed status=%d\n", event->connect.status);
            start_scan();
            return 0;
        }
        g_conn_handle = event->connect.conn_handle;

        int rc = ble_gattc_disc_svc_by_uuid(g_conn_handle, &g_svc_uuid.u,
                                            discover_svc_cb, NULL);
        if (rc != 0) {
            printf("# RX: service discovery failed\n");
            ble_gap_terminate(g_conn_handle, BLE_ERR_REM_USER_CONN_TERM);
        }
        return 0;
    }

    case BLE_GAP_EVENT_DISCONNECT:
        printf("# RX: disconnected reason=%d\n", event->disconnect.reason);
        start_scan();
        return 0;

    case BLE_GAP_EVENT_NOTIFY_RX: {
        if (event->notify_rx.om->om_len < sizeof(sample_t)) {
            printf("# RX: short notify len=%u\n",
                   (unsigned)event->notify_rx.om->om_len);
            return 0;
        }

        sample_t sample;
        os_mbuf_copydata(event->notify_rx.om, 0, sizeof(sample), &sample);

        printf("%u,%d,%d,%d,%d,%d,%d\n",
               sample.seq,
               sample.temp_val, sample.temp_scale,
               sample.hum_val, sample.hum_scale,
               sample.press_val, sample.press_scale);

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
            printf("# RX: found %.*s, connecting...\n",
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
    const struct ble_gap_disc_params scan_params = { 10000, 200, 0, 0, 0, 1 };
    int rc = ble_gap_disc(g_addr_type, 100, &scan_params, scan_event, NULL);
    if (rc != 0) {
        printf("# RX: scan failed rc=%d\n", rc);
    }
}

int main(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    rc = ble_hs_id_infer_auto(0, &g_addr_type);
    assert(rc == 0);

    printf("seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale\n");

    start_scan();

    while (1) {
        ztimer_sleep(ZTIMER_MSEC, 1000);
    }

    return 0;
}
