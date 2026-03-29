# src/prepare_all_latency_data.py
import os
import numpy as np
import subprocess
import sys

TASKS = ["node", "env"]
SEQ_LENS = [100, 500, 1000]
OVERLAPS = [0.4, 0.5]

os.makedirs("data/processed", exist_ok=True)

for task in TASKS:
    for seq_len in SEQ_LENS:
        for overlap in OVERLAPS:
            print(f"\n--- Preparing latency dataset: task={task}, seq_len={seq_len}, overlap={overlap} ---")
            
            cmd = [
                sys.executable, "src/prepare_latency_data.py",
                "--task", task,
                "--seq_len", str(seq_len),
                "--overlap", str(overlap)
            ]
            
            subprocess.run(cmd)

print("\nDone preparing all latency datasets.")
