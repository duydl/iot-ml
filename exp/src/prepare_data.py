# src/prepare_data.py
import os
import argparse
import numpy as np
import pandas as pd

FILES = [
    "data/raw/e0-bridge.csv",
    "data/raw/e1-lake.csv",
    "data/raw/e2-forest.csv",
    "data/raw/e3-river.csv",
    "data/raw/e4-garden.csv"
]

DEVICE_TO_LABEL = {
    "RIOT-BLE-0": 0,
    "RIOT-BLE-1": 1,
    "RIOT-BLE-2": 2,
    "RIOT-BLE-3": 3
}

def create_dataset(task="node", seq_len=100, overlap=0.5):
    stride = int(seq_len * (1 - overlap))

    X, y, env_ids,node_ids = [], [], [],[]

    for env_id, file in enumerate(FILES):
        df = pd.read_csv(file)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

        for device, node_label in DEVICE_TO_LABEL.items():
            df_node = df[df["device"] == device].copy()
            if len(df_node) < seq_len:
                continue

            df_node["rssi_diff"] = df_node["rssi"].diff()
            df_node = df_node.dropna(subset=["rssi_diff"])

            y_min = df_node["rssi_diff"].min()
            y_max = df_node["rssi_diff"].max()
            if y_max - y_min == 0:
                continue

            df_node["rssi_norm"] = (df_node["rssi_diff"] - y_min) / (y_max - y_min)
            data = df_node["rssi_norm"].values

            for i in range(0, len(data) - seq_len + 1, stride):
                seq = data[i:i+seq_len]

                if task == "node":
                    label = node_label
                elif task == "env":
                    label = env_id
                else:
                    raise ValueError("task must be 'node' or 'env'")

                X.append(seq)
                y.append(label)
                env_ids.append(env_id)
                node_ids.append(node_label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    env_ids = np.array(env_ids, dtype=np.int64)
    node_ids = np.array(node_ids, dtype=np.int64)
    return X, y, env_ids,node_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["node", "env"])
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    args = parser.parse_args()

    X, y, env_ids ,node_ids= create_dataset(
        task=args.task,
        seq_len=args.seq_len,
        overlap=args.overlap
    )

    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/{args.task}_seq{args.seq_len}_ov{int(args.overlap*100)}.npz"

    np.savez(out_path, X=X, y=y, env_ids=env_ids,node_ids=node_ids)

    print("Saved to:", out_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("env_ids shape:", env_ids.shape)
    print("node_ids shape:", node_ids.shape)


if __name__ == "__main__":
    main()