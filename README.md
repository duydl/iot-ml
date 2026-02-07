# IoT / ML Project

## IoT Setup

1. **Build and flash TX/RX** (two boards):
```bash
make -C iot/tx flash
make -C iot/rx flash
```

2. **Log RX CSV from `_Project` root** (creates `data/<timestamp>/rx.csv` using pyterm timestamps):
```bash
./iot/log_rx.sh
```
CSV header:
```
ts,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale,rssi
```
`ts` is the pyterm timestamp prefix. Set `PORT=/dev/ttyACM1` if needed.

## ML Setup

1. **Install uv**:
   [Follow the instructions here](https://github.com/astral-sh/uv) or install via pip:
   ```bash
   pip install uv
   ```

2. **Install Dependencies**:
   Run the following command in the project root (`_Project/`):

```bash
uv sync
```

1. **Run Training**:
   Run the training script from the project root:
```bash
uv run python ml/train.py
```
