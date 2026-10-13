"""
train_cnn.py
============

Command-line entry point for Part 2 of the workshop: train a CNN on the
patch dataset produced by ``main.py`` (Part 1), then evaluate it on the
held-out test set.

Usage
-----
    python train_cnn.py --config configs/train.yaml

Requires PyTorch: ``pip install torch``.
"""

import argparse

from models.evaluate import (
    evaluate_on_test_set,
    load_trained_model,
    print_metrics_report,
)
from models.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN to detect RFI in MeerKAT amplitude patches."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)

    print(f"Training on device: {config.resolve_device()}")
    train(config)

    print("\nEvaluating best model on the held-out test set...")
    best_checkpoint = f"{config.checkpoint_dir}/best_model.pt"
    model = load_trained_model(best_checkpoint)
    metrics = evaluate_on_test_set(model, config.dataset_dir)
    print_metrics_report(metrics)


if __name__ == "__main__":
    main()
