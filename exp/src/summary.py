import os
import json
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

OUTPUT_DIR = "outputs"
RESULTS_DIR = "outputs/results"


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
    if HAS_SEABORN:
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
    if HAS_SEABORN:
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
    for task in sorted(df["task"].dropna().unique()):
        for split in sorted(df["split"].dropna().unique()):
            subset = df[(df["task"] == task) & (df["split"] == split)]
            if subset.empty:
                continue

            subset = subset.copy()
            subset["setting"] = subset["model"] + "_ov" + (subset["overlap"] * 100).astype(int).astype(str)

            pivot = subset.pivot_table(
                index="seq_len",
                columns="setting",
                values="test_acc",
                aggfunc="mean",
            )

            plt.figure(figsize=(10, 4))
            if HAS_SEABORN:
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
            plt.title(f"Test Accuracy Heatmap ({task}, {split})")
            plt.xlabel("Model + Overlap")
            plt.ylabel("Sequence Length")
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, f"heatmap_acc_{task}_{split}.png"),
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
    if HAS_SEABORN:
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