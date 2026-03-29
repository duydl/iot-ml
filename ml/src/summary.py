import os
import json
import re
import pandas as pd
from typing import Any
from models.cnn import CNN1D
from models.resnet import ResNet1D
from utils.plotting import (
    plot_raw_data_observation,
    plot_model_split_bar,
    plot_seq_overlap_heatmaps,
    plot_top_experiments
)

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