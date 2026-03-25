# src/prepare_all_data.py
import os
import numpy as np
from prepare_data import create_dataset

TASKS = ["node", "env"]
SEQ_LENS = [100, 500, 1000]
OVERLAPS = [0.4, 0.5]

os.makedirs("data/processed", exist_ok=True)

for task in TASKS:
    for seq_len in SEQ_LENS:
        for overlap in OVERLAPS:
            print(f"Preparing: task={task}, seq_len={seq_len}, overlap={overlap}")

            X, y, env_ids,node_ids = create_dataset(
                task=task,
                seq_len=seq_len,
                overlap=overlap
            )

            out_path = f"data/processed/{task}_seq{seq_len}_ov{int(overlap*100)}.npz"
            np.savez(out_path, X=X, y=y, env_ids=env_ids,node_ids=node_ids)

            print(f"Saved to {out_path}, X shape={X.shape}")