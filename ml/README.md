# ML Module: Node/Environment Classification

This module handles data preparation, model training, and evaluation for RSSI-based localization and environment recognition.

## Setup

1. **Install uv**:
   Follow instructions [here](https://github.com/astral-sh/uv) or install via pip:
   ```bash
   pip install uv
   ```

2. **Sync Dependencies**:
   From the project root (`_Project/`), sync the environment including the `ml` extra:
   ```bash
   uv sync --extra ml
   ```

## Workflow

1. **Data Preprocessing**:
   Convert raw CSV data into training-ready `.npz` files.
   ```bash
   uv run python ml/src/prepare_all_data.py
   ```

2. **Model Training**:
   Run an experiment with custom parameters.
   ```bash
   uv run python ml/src/run_experiment.py --task node --seq_len 100 --overlap 0.5 --split random --model cnn
   ```

## Structure

- `data/raw/`: raw CSV files (`e0-bridge.csv` ... `e4-garden.csv`)
- `data/processed/`: processed `.npz` datasets
- `outputs/`: experiment outputs (metrics, plots, checkpoints)
- `src/`: training / preprocessing / analysis scripts

## Key Scripts

- `src/prepare_data.py`: Build one processed dataset.
- `src/prepare_all_data.py`: Build multiple datasets in batch.
- `src/run_experiment.py`: Run one experiment.
- `src/run_all_exp.py`: Run batch experiments.
- `src/summary.py`: Aggregate metrics and generate plots.
- `src/splitbytime.ipynb`: Split train/test based on time instead of random splitting.

## Detailed Methodology

1. **Prepare datasets**:
    - Load each environment CSV, sort by timestamp, and process each device stream separately.
    - Convert RSSI to `rssi_diff` and apply per-device min-max normalization.
    - Generate overlapping windows (`seq_len`, `overlap`).
    - Assign labels (`node` or `env`) and save `X`, `y`, `env_ids`, and `node_ids` into `data/processed/*.npz`.
2. **Run experiments**: Train models (`cnn`, `resnet`) on the processed datasets.
3. **Analyze**: Aggregate results across all configurations and visualize performance.

## Example Commands

Run these from the project root (`_Project/`) using `uv run`.

### 1) Prepare all datasets
```bash
uv run --extra ml python ml/src/prepare_all_data.py
```

### 2) Run one node classification experiment
```bash
uv run --extra ml python ml/src/run_experiment.py \
    --task node \
    --seq_len 100 \
    --overlap 0.5 \
    --split random \
    --model cnn \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

Run from the `exp/` directory.

### 1) Prepare one dataset
```bash
python src/prepare_data.py --task node --seq_len 100 --overlap 0.5
```

## Output Files (per experiment)

Each experiment folder under `outputs/` usually includes:

- `metrics.json`: Summary of experiment settings and results.
- `classification_report.txt`: F1-score, precision, and recall per class.
- `confusion_matrix.npy`: Raw confusion matrix data.
- `confusion_matrix.png`: Visual confusion matrix plot.
- `training_curves.png`: Training/test loss and accuracy curves.
- `best_model.pt`: Weights of the best performing model.

## Experiments Configrations

- **Models**: `cnn`, `resnet`
- **Splits**: `random`, `oneout`
- **Overlaps**: `0.4`, `0.5`
- **Seq Lengths**: `100`, `500`, `1000`
- **Tasks**: `node` (classification of device ID), `env` (classification of environment)

Total: `2 × 2 × 2 × 3 × 2 = 48` experimental configurations.

## Notes
- `task=node`: classify node ID (`RIOT-BLE-0` to `RIOT-BLE-3`)
- `task=env`: classify environment (`e0` to `e4`)
- `split=oneout`: For node recognition, use `env3` as the test set; for environment recognition, use `node1` as the test set.
- `splitbytime` performs slightly worse than random split (`0.69` vs `0.77` accuracy for `node_seq100_ov50_random_cnn`).
