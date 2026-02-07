#!/usr/bin/env python3
import argparse
import os

import pytorch_lightning as pl

from data import TimeSeriesDataModule
from model import TimeSeriesClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 1D CNN/ResNet on preprocessed time-series.")
    parser.add_argument("--data", default=None, help="Path to preprocessed .npz dataset")
    parser.add_argument("--task", choices=["env", "node"], default="env")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--split", choices=["random", "group_holdout"], default="random")
    parser.add_argument("--group-key", choices=["env", "node"], default=None,
                        help="Group key for group_holdout split")
    parser.add_argument("--holdout", default=None,
                        help="Holdout label name or index for group_holdout split")
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--test", action="store_true", help="Run test after training")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = args.data or os.path.join(repo_root, "ml", "processed", "dataset.npz")

    if args.split == "group_holdout" and args.group_key is None:
        args.group_key = "node" if args.task == "env" else "env"

    pl.seed_everything(args.seed)

    dm = TimeSeriesDataModule(
        data_path=data_path,
        task=args.task,
        batch_size=args.batch_size,
        split=args.split,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        group_key=args.group_key,
        holdout=args.holdout,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup("fit")

    model = TimeSeriesClassifier(
        in_channels=dm.input_channels,
        num_classes=dm.num_classes,
        arch=args.model,
        learning_rate=args.lr,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
    )

    trainer.fit(model, datamodule=dm)

    if args.test:
        trainer.test(model, datamodule=dm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
