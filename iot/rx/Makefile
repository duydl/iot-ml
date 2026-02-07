# name of your application
APPLICATION = iot_rx

# If no BOARD is found in the environment, use this default:
BOARD ?= adafruit-feather-nrf52840-sense
PROGRAMMER = adafruit-nrfutil
# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= $(CURDIR)/../../RIOT

# Some RIOT modules needed for this example
USEMODULE += ztimer_msec

# Include NimBLE
USEMODULE += nimble_svc_gap
USEMODULE += nimble_svc_gatt

# Allow multiple simultaneous BLE connections
RX_MAX_CONN ?= 4
CFLAGS += -DMYNEWT_VAL_BLE_MAX_CONNECTIONS=$(RX_MAX_CONN)
CFLAGS += -DRX_MAX_CONN=$(RX_MAX_CONN)

# Comment this out to disable code in RIOT that does safety checking
# which is not needed in a production environment but helps in the
# development process:
DEVELHELP ?= 1

# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

include $(RIOTBASE)/Makefile.include
