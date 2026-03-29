import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Optional

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

def set_plot_style():
    """
    Sets a standard, paper-appropriate style for all plots.
    """
    # Use a professional style if seaborn is available
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

# Initialize style on module load
set_plot_style()

def plot_raw_data_observation(raw_df: pd.DataFrame, results_dir: str) -> None:
    if raw_df.empty:
        print("No raw CSV data found.")
        return

    os.makedirs(results_dir, exist_ok=True)

    if "env_name" in raw_df.columns:
        env_counts = raw_df.groupby("env_name").size().sort_index()
        env_values = env_counts.tolist()
        env_labels = [f"{env} ({count})" for env, count in env_counts.items()]
        plt.figure(figsize=(6, 6))
        plt.pie(env_values, labels=env_labels, autopct="%.1f%%", startangle=140)
        plt.title("Environment Portion (Count + Ratio)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "raw_portion_env.png"))
        plt.close()

    if "device" in raw_df.columns:
        expected_nodes = [f"RIOT-BLE-{index}" for index in range(4)]
        node_counts = raw_df["device"].value_counts().reindex(expected_nodes, fill_value=0)
        node_values = node_counts.tolist()
        node_labels = [
            f"n{str(device).split('-')[-1]} ({count})"
            for device, count in node_counts.items()
        ]
        plt.figure(figsize=(6, 6))
        plt.pie(node_values, labels=node_labels, autopct="%.1f%%", startangle=140)
        plt.title("Node Portion (Count + Ratio)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "raw_portion_node.png"))
        plt.close()

    if "rssi" in raw_df.columns and "env_name" in raw_df.columns:
        rssi_df = raw_df[["env_name", "rssi"]].copy()
        rssi_df["rssi"] = pd.to_numeric(rssi_df["rssi"], errors="coerce")
        rssi_df = rssi_df.dropna(subset=["rssi"])

        if not rssi_df.empty:
            env_order = sorted(rssi_df["env_name"].unique().tolist())
            rssi_by_env: List[List[float]] = []
            for env_name in env_order:
                env_frame = rssi_df[rssi_df["env_name"] == env_name]
                env_rssi = [float(value) for value in env_frame["rssi"].tolist()]
                rssi_by_env.append(env_rssi)

            plt.figure(figsize=(10, 5))
            plt.boxplot(rssi_by_env, tick_labels=env_order, showfliers=False)

            for x_pos, env_rssi in enumerate(rssi_by_env, start=1):
                if not env_rssi: continue
                min_value = min(env_rssi)
                max_value = max(env_rssi)
                plt.text(x_pos + 0.05, max_value, f"max={max_value:.0f}", fontsize=8, va="bottom")
                plt.text(x_pos + 0.05, min_value, f"min={min_value:.0f}", fontsize=8, va="top")

            plt.title("RSSI Distribution by Environment")
            plt.xlabel("Environment")
            plt.ylabel("RSSI (dBm)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "raw_rssi_min_max_by_env.png"))
            plt.close()

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
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_mean_acc_task_model.png"))
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
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_mean_f1_task_model.png"))
    plt.close()

def plot_seq_overlap_heatmaps(df: pd.DataFrame, results_dir: str) -> None:
    for model in sorted(df["model"].dropna().unique()):
        for task in sorted(df["task"].dropna().unique()):
            subset = df[(df["model"] == model) & (df["task"] == task)].copy()
            if subset.empty: continue

            subset["setting"] = subset["split"] + "_ov" + (subset["overlap"] * 100).astype(int).astype(str)
            pivot = subset.pivot_table(index="seq_len", columns="setting", values="test_acc", aggfunc="mean")

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
            
            plt.title(f"Test Accuracy Heatmap: {model} model on {task} task")
            plt.xlabel("Split & Overlap")
            plt.ylabel("Seq Length")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"heatmap_acc_{model}_{task}.png"))
            plt.close()

def plot_top_experiments(df: pd.DataFrame, results_dir: str, top_k: int = 15) -> None:
    top_df = df.sort_values("test_acc", ascending=False).head(top_k).copy()
    if top_df.empty: return

    top_df["label"] = (
        top_df["task"] + "|" + top_df["model"] + "|" + top_df["split"] + 
        "|L" + top_df["seq_len"].astype(str) + "|ov" + (top_df["overlap"] * 100).astype(int).astype(str)
    )

    plt.figure(figsize=(12, 6))
    if HAS_SEABORN:
        sns.barplot(data=top_df, y="label", x="test_acc", hue="model", dodge=False)
    else:
        plt.barh(top_df["label"], top_df["test_acc"])
    
    plt.title(f"Top {top_k} Experiments by Test Accuracy")
    plt.xlabel("Test Accuracy")
    plt.ylabel("Experiment Config")
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_top_experiments_test_acc.png"))
    plt.close()

def plot_training_curves(train_losses: List[float], test_losses: List[float], 
                         train_accs: List[float], test_accs: List[float], 
                         output_dir: Optional[str] = None) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.legend()

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm: np.ndarray, output_dir: Optional[str] = None, class_names: Optional[List[str]] = None) -> None:
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names if class_names else "auto",
                    yticklabels=class_names if class_names else "auto")
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names)) if class_names else np.arange(cm.shape[0])
        plt.xticks(tick_marks, class_names if class_names else tick_marks, rotation=45)
        plt.yticks(tick_marks, class_names if class_names else tick_marks)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    else:
        plt.show()
