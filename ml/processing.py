# ========== To Do ==========
# different frame length
# discuss what type of frames output format is needed
# different label for different enviroments (maybe rename csv files?)

import pandas as pd
import numpy as np

DEVICES = [
    "RIOT-BLE-0",
    "RIOT-BLE-1",
    "RIOT-BLE-2",
    "RIOT-BLE-3"
]

folders = [
    "data/20260306_155237_bridge1",
    "data/20260306_162334_bridge2",
    # "data/20260306_175404_lake1",
    # "data/20260307_144319_forest",
    # "data/20260307_162641_river",
    # "data/20260307_173911_garden"
]

def clean_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    df["environment"] = "bridge"

    # data type
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce")

    # drop missing values
    df = df.dropna(subset=["ts", "device", "rssi"])

    # filter devices
    df = df[df["device"].isin(DEVICES)]

    # sort by device, timestamp, seq
    df = df.sort_values(["device", "ts", "seq"])

    # filter invalid rssi
    df = df[(df["rssi"] >= -110) & (df["rssi"] <= -20)]

    return df


def minmax_normalize(y: np.ndarray) -> np.ndarray:
    y_min = y.min()
    y_max = y.max()

    if y_max == y_min:
        return np.zeros_like(y, dtype=np.float32)

    return ((y - y_min) / (y_max - y_min)).astype(np.float32)


def sliding_windows(signal: np.ndarray, frame_size: int, overlap: float) -> np.ndarray:
    step = int(frame_size * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too much, step <= 0")

    frames = []
    for start in range(0, len(signal) - frame_size + 1, step):
        end = start + frame_size
        frames.append(signal[start:end])

    if len(frames) == 0:
        return np.empty((0, frame_size), dtype=np.float32)

    return np.stack(frames).astype(np.float32)


def preprocess(csv_path: str, frame_size: int = 100, overlap: float = 0.5):
    df = clean_csv(f"{csv_path}/rx.csv")

    X_list = []
    y_list = []
    meta_list = []

    env = df["environment"].iloc[0] if len(df) > 0 else None

    for device, g in df.groupby("device", sort=False):
        rssi = g["rssi"].to_numpy(dtype=np.float32)

        if len(rssi) < 2:
            continue

        # y_i = x_{i+1} - x_i
        diff_rssi = np.diff(rssi)

        if len(diff_rssi) < frame_size:
            continue

        norm_rssi = minmax_normalize(diff_rssi)

        frames = sliding_windows(norm_rssi, frame_size, overlap)

        if len(frames) == 0:
            continue

        # csv for observation
        # if len(frames) > 0:
        #     pd.DataFrame(frames).to_csv(
        #         f"{csv_path}/frames_{device}.csv",
        #         index=False
        #     )

        X_list.append(frames)
        y_list.extend([device] * len(frames))
        meta_list.extend([
            {
                "environment": env,
                "device": device
            }
            for _ in range(len(frames))
        ])

    if not X_list:
        X = np.empty((0, 1, frame_size), dtype=np.float32)
        y = np.array([], dtype=object)
        meta = pd.DataFrame(columns=["environment", "device"])
        return X, y, meta

    X = np.vstack(X_list).astype(np.float32)
    X = np.expand_dims(X, axis=1)   # (N, 1, L)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_list)


    return X, y, meta


if __name__ == "__main__":
    for csv_path in folders:
        X, y, meta = preprocess(
            csv_path,
            frame_size=100,
            overlap=0.5
        )


