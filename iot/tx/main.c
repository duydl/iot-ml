/*
 * BLE TX (peripheral): read SHT humidity + BMP280 temperature/pressure via SAUL,
 * then notify the central at 10 Hz with raw phydat values (val + scale).
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
#include "saul_reg.h"

#define CUSTOM_SVC_UUID     0xff00
#define CUSTOM_CHR_UUID     0xee00
#define DEVICE_NAME         "RIOT-IOT-TX"

#define SHT_NAME            "sht3x1"
#define BMP_NAME            "bmp280"

#define SAMPLE_PERIOD_MS    100

typedef struct __attribute__((packed)) {
    uint16_t seq;
    int16_t temp_val;
    int8_t temp_scale;
    int16_t hum_val;
    int8_t hum_scale;
    int16_t press_val;
    int8_t press_scale;
} sample_t;

static uint8_t g_addr_type;
static uint8_t g_conn_state;
static uint8_t g_notify_state;
static uint16_t g_conn_handle;
static uint16_t g_notify_val_handle;

static saul_reg_t *g_temp_dev;
static saul_reg_t *g_hum_dev;
static saul_reg_t *g_press_dev;
static uint8_t g_sensors_ready;

static void start_advertising(void);

static saul_reg_t *find_dev(uint8_t type, const char *name, const char *label)
{
    saul_reg_t *dev = saul_reg_find_type_and_name(type, name);
    if (!dev) {
        printf("# TX: missing %s (name=%s, type=%u)\n", label, name, type);
    }
    return dev;
}

static int gatt_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                          struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;

    if (ble_uuid_u16(ctxt->chr->uuid) != CUSTOM_CHR_UUID) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    return 0;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID16_DECLARE(CUSTOM_SVC_UUID),
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = BLE_UUID16_DECLARE(CUSTOM_CHR_UUID),
                .access_cb = gatt_access_cb,
                .val_handle = &g_notify_val_handle,
                .flags = BLE_GATT_CHR_F_NOTIFY,
            },
            { 0 },
        },
    },
    { 0 },
};

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_ADV_COMPLETE:
        start_advertising();
        return 0;

    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status != 0) {
            printf("# TX: connect failed status=%d\n", event->connect.status);
            g_conn_state = 0;
            g_notify_state = 0;
            start_advertising();
            return 0;
        }
        g_conn_state = 1;
        g_notify_state = 0;
        g_conn_handle = event->connect.conn_handle;
        printf("# TX: connected handle=%u\n", g_conn_handle);
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        printf("# TX: disconnected reason=%d\n", event->disconnect.reason);
        g_conn_state = 0;
        g_notify_state = 0;
        start_advertising();
        return 0;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == g_notify_val_handle) {
            g_notify_state = event->subscribe.cur_notify;
            printf("# TX: notify_state=%u\n", g_notify_state);
        }
        return 0;

    case BLE_GAP_EVENT_NOTIFY_TX:
        return 0;
    }

    return 0;
}

static void start_advertising(void)
{
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;

    memset(&adv_params, 0, sizeof(adv_params));
    memset(&fields, 0, sizeof(fields));

    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    fields.flags = BLE_HS_ADV_F_DISC_GEN;
    fields.name = (uint8_t *)DEVICE_NAME;
    fields.name_len = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;

    fields.uuids16 = (ble_uuid16_t[]) { BLE_UUID16_INIT(CUSTOM_SVC_UUID) };
    fields.num_uuids16 = 1;
    fields.uuids16_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        printf("# TX: adv_set_fields failed rc=%d\n", rc);
        return;
    }

    rc = ble_gap_adv_start(g_addr_type, NULL, 10000,
                           &adv_params, gap_event, NULL);
    if (rc != 0) {
        printf("# TX: adv_start failed rc=%d\n", rc);
    } else {
        printf("# TX: advertising\n");
    }
}

int main(void)
{
    int rc = ble_svc_gap_device_name_set(DEVICE_NAME);
    assert(rc == 0);

    rc = ble_gatts_count_cfg(gatt_svcs);
    assert(rc == 0);
    rc = ble_gatts_add_svcs(gatt_svcs);
    assert(rc == 0);
    rc = ble_gatts_start();
    assert(rc == 0);

    g_temp_dev = find_dev(SAUL_SENSE_TEMP, BMP_NAME, "bmp280 temp");
    g_press_dev = find_dev(SAUL_SENSE_PRESS, BMP_NAME, "bmp280 press");
    g_hum_dev = find_dev(SAUL_SENSE_HUM, SHT_NAME, "sht3x hum");
    g_sensors_ready = (g_temp_dev && g_press_dev && g_hum_dev);

    rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    rc = ble_hs_id_infer_auto(0, &g_addr_type);
    assert(rc == 0);

    start_advertising();

    uint16_t seq = 0;
    while (1) {
        if (g_conn_state && g_notify_state && g_sensors_ready) {
            phydat_t temp;
            phydat_t hum;
            phydat_t press;

            if (saul_reg_read(g_temp_dev, &temp) < 1) {
                printf("# TX: temp read failed\n");
                goto sleep;
            }
            if (saul_reg_read(g_hum_dev, &hum) < 1) {
                printf("# TX: hum read failed\n");
                goto sleep;
            }
            if (saul_reg_read(g_press_dev, &press) < 1) {
                printf("# TX: press read failed\n");
                goto sleep;
            }

            sample_t sample = {
                .seq = seq++,
                .temp_val = temp.val[0],
                .temp_scale = temp.scale,
                .hum_val = hum.val[0],
                .hum_scale = hum.scale,
                .press_val = press.val[0],
                .press_scale = press.scale,
            };

            struct os_mbuf *om = ble_hs_mbuf_from_flat(&sample, sizeof(sample));
            if (om == NULL) {
                printf("# TX: mbuf alloc failed\n");
                goto sleep;
            }

            int rc = ble_gatts_notify_custom(g_conn_handle, g_notify_val_handle, om);
            if (rc != 0) {
                printf("# TX: notify failed rc=%d\n", rc);
                os_mbuf_free_chain(om);
            }
        }

sleep:
        ztimer_sleep(ZTIMER_MSEC, SAMPLE_PERIOD_MS);
    }

    return 0;
}
