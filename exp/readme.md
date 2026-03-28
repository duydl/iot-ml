# EXP Module (IoT Node/Environment Classification)

## Structure

- `data/raw/`: raw CSV files (`e0-bridge.csv` ... `e4-garden.csv`)
- `data/processed/`: processed `.npz` datasets
- `outputs/`: experiment outputs (metrics, plots, checkpoints)
- `src/`: training / preprocessing / analysis scripts

## Workflow

1. Prepare datasets:
    1.1 load each environment CSV, sort by timestamp, and process each device stream separately. 
    1.2 converts RSSI to `rssi_diff`, applies per-device min-max normalization
    1.3 generates overlapping windows (`seq_len`, `overlap`). 
    1.4 Finally, it assigns labels (`node` or `env`) and saves `X`, `y`, and `env_ids`,`node_ids` into `data/processed/*.npz`.
2. Run experiments (single run or batch)
3. Summarize and visualize results


## Key Scripts

- `src/prepare_data.py`: build one processed dataset
- `src/prepare_all_data.py`: build multiple datasets in batch
- `src/run_experiment.py`: run one experiment
- `src/run_all_exp.py`: run batch experiments
- `src/summary.py`: aggregate metrics and generate plots
- `src/splitbytime.ipynb`: split train/test based on time instead of random splitting
- `outputs/results`: includes summary tables and visualization figures across all experiments.  
## Example Commands

Run from the `exp/` directory.

### 1) Prepare one dataset

```bash
python src/prepare_data.py --task node --seq_len 100 --overlap 0.5
```

### 2) Run one experiment

```bash
python src/run_experiment.py \
	--task node \
	--seq_len 100 \
	--overlap 0.5 \
	--split random \
	--model cnn \
	--epochs 100 \
	--batch_size 64 \
	--lr 1e-4
```

## Output Files (per experiment)

Each experiment folder under `outputs/` usually includes:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.npy`
- `confusion_matrix.png`
- `training_curves.png`
- `best_model.pt`

## Experiments
- 2 models: `cnn`, `resnet`
- 2 splits: `random`, `oneout`
- 2 overlaps: `0.4`, `0.5`
- 3 sequence lengths: `100`, `500`, `1000`
- 2 tasks (objects): `node`, `env`

Total:`2 × 2 × 2 × 3 × 2 = 48` experiments.
## Notes
- `task=node`: classify node ID (`RIOT-BLE-0` to `RIOT-BLE-3`)
- `task=env`: classify environment (`e0` to `e4`)
- `split=oneout`: for node recognition, use `env3` as the test set and the other environments as the training set; for environment recognition, use `node1` as the test set and the other nodes as the training set.
- `splitbytime` performs slightly worse than random split; with the same configuration (`node_seq100_ov50_random_cnn`), accuracy is `0.69` for split-by-time vs `0.77` for random split.

random split t1 t2-> train: t1, test: t2