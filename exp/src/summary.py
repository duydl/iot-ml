import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from models.cnn import CNN1D
from models.resnet import ResNet1D

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

OUTPUT_DIR = "outputs"
RESULTS_DIR = "outputs/results"
RAW_DATA_DIR = "data/raw"

# for model architecture summary and parameter count
def model_arch(model: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    arch_path = os.path.join(output_dir, "model_arch.txt")
    with open(arch_path, "w") as f:
        f.write(str(model))

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    params_path = os.path.join(output_dir, "model_params.json")
    with open(params_path, "w") as f:
        json.dump(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            f,
            indent=2,
        )
def save_default_model_architectures(results_dir: str, num_classes: int = 4) -> None:
    model_root = os.path.join(results_dir, "model")
    models = {
        "cnn": CNN1D(num_classes=num_classes),
        "resnet": ResNet1D(num_classes=num_classes),
    }

    for model_name, model in models.items():
        model_output_dir = os.path.join(model_root, model_name)
        model_arch(model, output_dir=model_output_dir)
        print(f"Saved model info: {model_name} -> {model_output_dir}")

# for raw data observation
def load_raw_data(raw_data_dir: str) -> pd.DataFrame:
    csv_paths = []
    for root, _, files in os.walk(raw_data_dir):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, file_name))

    frames = []
    for csv_path in sorted(csv_paths):
        try:
            raw_part = pd.read_csv(csv_path)
            file_name = os.path.basename(csv_path)
            env_match = re.search(r"(e\d+)", file_name.lower())
            env_name = env_match.group(1) if env_match else "unknown"
            raw_part["env_name"] = env_name
            raw_part["source_file"] = file_name
            frames.append(raw_part)
        except Exception as error:
            print(f"Skip unreadable CSV: {csv_path} ({error})")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
def plot_raw_data_observation(raw_df: pd.DataFrame, results_dir: str) -> None:
    if raw_df.empty:
        print(f"No raw CSV data found under {RAW_DATA_DIR}")
        return

    os.makedirs(results_dir, exist_ok=True)

    if "env_name" in raw_df.columns:
        env_counts = raw_df.groupby("env_name").size().sort_index()
        env_values = env_counts.tolist()
        env_labels = [f"{env} ({count})" for env, count in env_counts.items()]
        plt.figure(figsize=(6, 6))
        plt.pie(env_values, labels=env_labels, autopct="%.1f%%")
        plt.title("Environment Portion (Count + Ratio)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "raw_portion_env.png"), dpi=200)
        plt.close()
    else:
        print("Skip env portion plot: 'env_name' column not found")

    if "device" in raw_df.columns:
        expected_nodes = [f"RIOT-BLE-{index}" for index in range(4)]
        node_counts = raw_df["device"].value_counts().reindex(expected_nodes, fill_value=0)
        node_values = node_counts.tolist()
        node_labels = [
            f"n{str(device).split('-')[-1]} ({count})"
            for device, count in node_counts.items()
        ]
        plt.figure(figsize=(6, 6))
        plt.pie(node_values, labels=node_labels, autopct="%.1f%%")
        plt.title("Node Portion (Count + Ratio)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "raw_portion_node.png"), dpi=200)
        plt.close()
    else:
        print("Skip node portion plot: 'device' column not found")

    if "rssi" in raw_df.columns and "env_name" in raw_df.columns:
        rssi_df = raw_df[["env_name", "rssi"]].copy()
        rssi_df["rssi"] = pd.to_numeric(rssi_df["rssi"], errors="coerce")
        rssi_df = rssi_df.dropna(subset=["rssi"])

        if not rssi_df.empty:
            env_order = sorted(rssi_df["env_name"].unique().tolist())
            rssi_by_env: list[list[float]] = []
            for env_name in env_order:
                env_frame = rssi_df[rssi_df["env_name"] == env_name]
                env_rssi = [float(value) for value in env_frame["rssi"].tolist()]
                rssi_by_env.append(env_rssi)

            plt.figure(figsize=(10, 5))
            plt.boxplot(rssi_by_env, tick_labels=env_order, showfliers=False)

            for x_pos, env_rssi in enumerate(rssi_by_env, start=1):
                if not env_rssi:
                    continue
                min_value = min(env_rssi)
                max_value = max(env_rssi)
                plt.text(x_pos + 0.05, max_value, f"max={max_value:.0f}", fontsize=8, va="bottom")
                plt.text(x_pos + 0.05, min_value, f"min={min_value:.0f}", fontsize=8, va="top")

            plt.title("RSSI Min/Max Distribution by Environment")
            plt.xlabel("Environment")
            plt.ylabel("RSSI")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "raw_rssi_min_max_by_env.png"), dpi=200)
            plt.close()
        else:
            print("Skip rssi distribution: all values are NaN or non-numeric")
    else:
        print("Skip rssi distribution: 'rssi' or 'env_name' column not found")

# for model output summary
def collect_metrics(output_dir: str) -> pd.DataFrame:
    rows = []

    for exp_name in os.listdir(output_dir):
        exp_path = os.path.join(output_dir, exp_name)

        if not os.path.isdir(exp_path):
            continue

        metrics_path = os.path.join(exp_path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        rows.append(
            {
                "experiment": exp_name,
                "task": metrics.get("task"),
                "model": metrics.get("model"),
                "split": metrics.get("split"),
                "test_env": metrics.get("test_env"),
                "seq_len": metrics.get("seq_len"),
                "overlap": metrics.get("overlap"),
                "epochs": metrics.get("epochs"),
                "batch_size": metrics.get("batch_size"),
                "learning_rate": metrics.get("learning_rate"),
                "train_acc": metrics.get("final_train_acc"),
                "test_acc": metrics.get("final_test_acc"),
                "test_f1_macro": metrics.get("final_test_f1_macro"),
                "train_loss": metrics.get("final_train_loss"),
                "test_loss": metrics.get("final_test_loss"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["task", "model", "split", "seq_len", "overlap"]).reset_index(drop=True)
    return df
def save_summary_table(df: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")
def plot_model_split_bar(df: pd.DataFrame, results_dir: str) -> None:
    agg = (
        df.groupby(["task", "split", "model"], as_index=False)[["test_acc", "test_f1_macro"]]
        .mean()
        .sort_values(["task", "split", "model"])
    )

    plt.figure(figsize=(12, 5))
    if HAS_SEABORN and sns is not None:
        sns.barplot(data=agg, x="task", y="test_acc", hue="model", errorbar=None)
    else:
        bar_df = agg.groupby(["task", "model"], as_index=False)["test_acc"].mean()
        pivot = bar_df.pivot(index="task", columns="model", values="test_acc")
        pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Mean Test Accuracy by Task / Model")
    plt.xlabel("Task")
    plt.ylabel("Mean Test Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_mean_acc_task_model.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    if HAS_SEABORN and sns is not None:
        sns.barplot(data=agg, x="task", y="test_f1_macro", hue="model", errorbar=None)
    else:
        bar_df = agg.groupby(["task", "model"], as_index=False)["test_f1_macro"].mean()
        pivot = bar_df.pivot(index="task", columns="model", values="test_f1_macro")
        pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Mean Test Macro-F1 by Task / Model")
    plt.xlabel("Task")
    plt.ylabel("Mean Test Macro-F1")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_mean_f1_task_model.png"), dpi=200)
    plt.close()
def plot_seq_overlap_heatmaps(df: pd.DataFrame, results_dir: str) -> None:
    for model in sorted(df["model"].dropna().unique()):
        for task in sorted(df["task"].dropna().unique()):
            subset = df[(df["model"] ==model ) & (df["task"] == task)]
            if subset.empty:
                continue

            subset = subset.copy()
            subset["setting"] = subset["split"] + "_ov" + (subset["overlap"] * 100).astype(int).astype(str)

            pivot = subset.pivot_table(
                index="seq_len",
                columns="setting",
                values="test_acc",
                aggfunc="mean",
            )

            plt.figure(figsize=(10, 4))
            if HAS_SEABORN and sns is not None:
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
            else:
                values = pivot.values
                im = plt.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
                plt.yticks(range(len(pivot.index)), pivot.index)
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        if pd.notna(values[i, j]):
                            plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9)
            plt.title(f"Test Accuracy Heatmap ({model}, {task})")
            plt.xlabel("Model + Overlap")
            plt.ylabel("Sequence Length")
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, f"heatmap_acc_{model}_{task}.png"),
                dpi=200,
            )
            plt.close()
def plot_top_experiments(df: pd.DataFrame, results_dir: str, top_k: int = 15) -> None:
    top_df = df.sort_values("test_acc", ascending=False).head(top_k).copy()
    if top_df.empty:
        return

    top_df["label"] = (
        top_df["task"]
        + "|"
        + top_df["model"]
        + "|"
        + top_df["split"]
        + "|L"
        + top_df["seq_len"].astype(str)
        + "|ov"
        + (top_df["overlap"] * 100).astype(int).astype(str)
    )

    plt.figure(figsize=(12, 6))
    if HAS_SEABORN and sns is not None:
        sns.barplot(data=top_df, y="label", x="test_acc", hue="model", dodge=False)
    else:
        plt.barh(top_df["label"], top_df["test_acc"])
    plt.title(f"Top {top_k} Experiments by Test Accuracy")
    plt.xlabel("Test Accuracy")
    plt.ylabel("Experiment")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_top_experiments_test_acc.png"), dpi=200)
    plt.close()
def print_best_configs(df: pd.DataFrame) -> None:
    if df.empty:
        return

    best_idx = df.groupby(["task", "split"])["test_acc"].idxmax()
    best_df = df.loc[best_idx, ["task", "split", "model", "seq_len", "overlap", "test_acc", "test_f1_macro"]]
    best_df = best_df.sort_values(["task", "split"]).reset_index(drop=True)

    print("\n=== Best config per task/split (by test_acc) ===")
    print(best_df.to_string(index=False))

def main():
    # for raw data observation
    raw_df = load_raw_data(RAW_DATA_DIR)
    plot_raw_data_observation(raw_df, RESULTS_DIR)
    # for model architecture summary and parameter count
    save_default_model_architectures(RESULTS_DIR, num_classes=4)
    # for model output summary
    df = collect_metrics(OUTPUT_DIR)
    if df.empty:
        print(f"No metrics.json found under {OUTPUT_DIR}")
        return
    save_summary_table(df, RESULTS_DIR)
    plot_model_split_bar(df, RESULTS_DIR)
    plot_seq_overlap_heatmaps(df, RESULTS_DIR)
    plot_top_experiments(df, RESULTS_DIR, top_k=15)
    print_best_configs(df)

    print(f"\nTotal experiments summarized: {len(df)}")
    print(f"Figures saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()