# name of your application
APPLICATION = iot_tx

# If no BOARD is found in the environment, use this default:
BOARD ?= adafruit-feather-nrf52840-sense
PROGRAMMER = adafruit-nrfutil
# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= $(CURDIR)/../../../RIOT

# Some RIOT modules needed for this example
USEMODULE += ztimer_msec
USEMODULE += saul
USEMODULE += saul_default

# Sensor settings for ~10 Hz sampling (SHT3x periodic 10 mps, low repeatability)
CFLAGS += -DSHT3X_PARAM_MODE=SHT3X_PERIODIC_10_MPS
CFLAGS += -DSHT3X_PARAM_REPEAT=SHT3X_LOW

# Include NimBLE
USEMODULE += nimble_svc_gap
USEMODULE += nimble_svc_gatt

# Comment this out to disable code in RIOT that does safety checking
# which is not needed in a production environment but helps in the
# development process:
DEVELHELP ?= 1

# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

include $(RIOTBASE)/Makefile.include
