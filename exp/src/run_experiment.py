import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from utils.split import split_dataset
from models.cnn import CNN1D
from models.resnet import ResNet1D

import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["env", "node"])
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["random", "leave_one_env_out"])
    parser.add_argument("--test_env", type=int, default=None)
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet"])

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()


def load_processed_data(task, seq_len, overlap):
    overlap_str = int(overlap * 100)
    file_path = f"data/processed/{task}_seq{seq_len}_ov{overlap_str}.npz"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed file not found: {file_path}")

    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
    env_ids = data["env_ids"]

    return X, y, env_ids, file_path


def build_model(model_name, num_classes):
    if model_name == "cnn":
        return CNN1D(num_classes=num_classes)
    elif model_name == "resnet":
        return ResNet1D(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_true = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)

    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            outputs = model(xb)
            loss = criterion(outputs, yb)

            total_loss += loss.item() * xb.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")

    return avg_loss, acc, f1, np.array(all_true), np.array(all_preds)


def make_output_dir(args):
    if args.split == "random":
        exp_name = f"{args.task}_seq{args.seq_len}_ov{int(args.overlap*100)}_{args.split}_{args.model}"
    else:
        exp_name = f"{args.task}_seq{args.seq_len}_ov{int(args.overlap*100)}_{args.split}_env{args.test_env}_{args.model}"

    output_dir = os.path.join("outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(output_dir, args, train_history, test_metrics, y_true, y_pred, cm):
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = train_history
    test_loss, test_acc, test_f1 = test_metrics

    metrics = {
        "task": args.task,
        "seq_len": args.seq_len,
        "overlap": args.overlap,
        "split": args.split,
        "test_env": args.test_env,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_train_loss": train_loss_list[-1],
        "final_train_acc": train_acc_list[-1],
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "final_test_f1_macro": test_f1
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))

    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, output_dir, class_names=None):
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. load data
    X, y, env_ids, data_path = load_processed_data(args.task, args.seq_len, args.overlap)
    print(f"Loaded: {data_path}")
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    # 2. split
    X_train, X_test, y_train, y_test, env_train, env_test = split_dataset(
        X, y, env_ids,
        split_strategy=args.split,
        test_size=0.25,
        random_state=args.random_state,
        test_env=args.test_env
    )
    print("Unique train envs:", np.unique(env_train))
    print("Unique test envs :", np.unique(env_test))

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # 3. reshape for Conv1D => (batch, channel, length)
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 4. tensor + loader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. build model
    num_classes = len(np.unique(y))
    model = build_model(args.model, num_classes=num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # 監控 loss，越小越好
        factor=0.5,        # lr 乘上 0.5
        patience=3,        # 連續 3 個 epoch 沒改善就降
        min_lr=1e-6
    )
    # 6. training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_test_acc = -1.0
    output_dir = make_output_dir(args)
    best_model_path = os.path.join(output_dir, "best_model.pt")
    best_test_loss = float("inf") #for early stopping
    early_stop_counter = 0
    early_stop_patience = 8
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{args.epochs}] ",
            f"LR: {current_lr:.6f} | ",
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
        )

        # 存 best model：建議看 test_loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    # 7. final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(y_true, y_pred)

    save_results(
        output_dir=output_dir,
        args=args,
        train_history=(train_losses, train_accs, test_losses, test_accs),
        test_metrics=(test_loss, test_acc, test_f1),
        y_true=y_true,
        y_pred=y_pred,
        cm=cm
    )
    # save plots
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, output_dir)
    plot_confusion_matrix(cm, output_dir)
    print("\nFinal Test Accuracy:", test_acc)
    print("Final Test F1 Macro:", test_f1)
    print("Confusion Matrix:\n", cm)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()