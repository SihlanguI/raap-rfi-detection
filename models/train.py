"""
train.py
========

Teaching notes
--------------
This module implements a standard supervised-learning training loop:

    for each epoch:
        for each batch of (image, label) pairs:
            1. forward pass:   prediction = model(image)
            2. compute loss:   how wrong is the prediction?
            3. backward pass:  compute gradients of the loss
            4. optimizer step: nudge the model's weights to reduce the loss
        evaluate on the validation set (no weight updates!)
        save a checkpoint if this is the best model so far

We keep the loop itself simple and readable (this is a teaching project,
not a production training framework), while still including the pieces
students should see in any real training script: a train/validation
split, per-epoch metrics, and checkpointing of the best model.

Loss function
-------------
Because RFI pixels are usually a small minority of all pixels (class
imbalance), plain pixel-wise accuracy can be misleading (a model that
predicts "clean" everywhere might still score >90% accuracy). We
therefore report precision, recall, F1, and IoU (Intersection over
Union) in addition to accuracy, and offer both:

- **BCE loss** (binary cross-entropy): the standard choice, treats every
  pixel independently.
- **Dice loss**: directly optimizes for mask overlap (similar to IoU),
  which often works better under class imbalance.

Both are implemented so students can compare them directly.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .cnn import SimpleRFIUNet
from .patch_dataset import RFIPatchDataset


@dataclass
class TrainConfig:
    """
    Configuration for training the CNN, normally loaded from
    ``configs/train.yaml``.

    Attributes
    ----------
    dataset_dir : str
        Path to the dataset produced by Part 1 (contains metadata.csv).
    checkpoint_dir : str
        Directory where model checkpoints will be saved.
    base_channels : int
        Width of the first CNN layer (see ``cnn.SimpleRFIUNet``).
    epochs : int
        Number of full passes over the training set.
    batch_size : int
        Number of patches per training batch.
    learning_rate : float
        Step size for the Adam optimizer.
    loss_function : str
        Either "bce" or "dice".
    num_workers : int
        Number of background worker processes for data loading.
    device : str
        "cuda", "cpu", or "auto" (auto picks cuda if available).
    seed : int
        Random seed, for reproducibility.
    """

    dataset_dir: str = "dataset"
    checkpoint_dir: str = "checkpoints"
    base_channels: int = 16
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    loss_function: str = "bce"
    num_workers: int = 2
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.

    Dice coefficient = 2 * |intersection| / (|pred| + |target|), which
    measures mask overlap directly (1.0 = perfect overlap). The loss is
    ``1 - dice`` so that minimizing the loss maximizes overlap.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs (pre-sigmoid), shape (B, 1, H, W).
    targets : torch.Tensor
        Ground-truth binary masks, same shape.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def get_loss_function(name: str):
    """Return a loss function given its config-file name ("bce" or "dice")."""
    name = name.lower().strip()
    if name == "bce":
        bce = nn.BCEWithLogitsLoss()
        return lambda logits, targets: bce(logits, targets)
    if name == "dice":
        return dice_loss
    raise ValueError(f"Unknown loss_function '{name}'. Use 'bce' or 'dice'.")


@torch.no_grad()
def compute_segmentation_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-wise classification metrics for a batch of predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs (pre-sigmoid), shape (B, 1, H, W).
    targets : torch.Tensor
        Ground-truth binary masks, same shape.
    threshold : float
        Probability threshold above which a pixel is predicted as RFI.

    Returns
    -------
    dict
        Dictionary with keys "accuracy", "precision", "recall", "f1", "iou".
        All values are plain Python floats.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _average_metrics(metric_dicts):
    """Average a list of per-batch metric dicts into a single dict."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: sum(d[k] for d in metric_dicts) / len(metric_dicts) for k in keys}


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """
    Run one epoch of either training (if ``optimizer`` is given) or
    evaluation (if ``optimizer`` is None) over ``dataloader``.

    Returns
    -------
    dict
        Averaged metrics for the epoch, including "loss" plus everything
        from :func:`compute_segmentation_metrics`.
    """
    is_training = optimizer is not None
    model.train(is_training)

    batch_metrics = []
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        batch_metrics.append(compute_segmentation_metrics(logits.detach(), labels))

    averaged = _average_metrics(batch_metrics)
    averaged["loss"] = total_loss / max(1, len(dataloader))
    return averaged


def train(config: TrainConfig) -> Dict[str, list]:
    """
    Full training loop: build the datasets/model/optimizer, then train
    for ``config.epochs`` epochs, saving the best-validation-F1 model to
    ``config.checkpoint_dir/best_model.pt``.

    Parameters
    ----------
    config : TrainConfig

    Returns
    -------
    dict
        A "history" dictionary with one list per tracked metric, e.g.
        ``history["train_loss"]``, ``history["val_f1"]``, etc. Handy for
        plotting training curves afterwards.
    """
    torch.manual_seed(config.seed)
    device = config.resolve_device()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    train_dataset = RFIPatchDataset(config.dataset_dir, split="train")
    val_dataset = RFIPatchDataset(config.dataset_dir, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SimpleRFIUNet(in_channels=1, base_channels=config.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = get_loss_function(config.loss_function)

    history: Dict[str, list] = {
        "train_loss": [], "train_f1": [], "train_iou": [],
        "val_loss": [], "val_f1": [], "val_iou": [],
    }

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    for epoch in range(1, config.epochs + 1):
        start = time.time()

        train_metrics = run_one_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_one_epoch(model, val_loader, loss_fn, device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_iou"].append(val_metrics["iou"])

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{config.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                },
                best_checkpoint_path,
            )
            print(f"  -> New best model saved (val_f1={best_val_f1:.4f})")

    print(f"Training complete. Best validation F1: {best_val_f1:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    return history
