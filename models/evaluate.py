"""
evaluate.py
===========

Teaching notes
--------------
Evaluation happens on the **test set**: patches the model never saw
during training or model-selection (validation). This gives an honest
estimate of how well the model would perform on new, unseen MeerKAT
data.

This module does two things:

1. Compute aggregate metrics (accuracy, precision, recall, F1, IoU) over
   the entire test set, plus a simple confusion matrix.
2. Visualize a handful of individual predictions - amplitude patch,
   ground-truth mask, predicted mask, and an overlay - so students can
   *look* at where the model succeeds and fails, not just read numbers.
   This qualitative inspection is exactly the kind of sanity-check a
   working radio astronomer would do before trusting a model's output.
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.visualization import plot_mask, plot_overlay, plot_waterfall
from .cnn import SimpleRFIUNet
from .patch_dataset import RFIPatchDataset
from .train import compute_segmentation_metrics, _average_metrics


def load_trained_model(
    checkpoint_path: str, device: Optional[torch.device] = None
) -> SimpleRFIUNet:
    """
    Reload a model from a checkpoint saved by ``train.train``.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pt`` checkpoint file (e.g. "checkpoints/best_model.pt").
    device : torch.device, optional
        Device to load the model onto. Defaults to CUDA if available.

    Returns
    -------
    SimpleRFIUNet
        The model, in evaluation mode, with trained weights loaded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = SimpleRFIUNet(in_channels=1, base_channels=config["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(validation F1 = {checkpoint['val_f1']:.4f})"
    )
    return model


@torch.no_grad()
def evaluate_on_test_set(
    model: SimpleRFIUNet,
    dataset_dir: str,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute averaged metrics over the entire test split.

    Parameters
    ----------
    model : SimpleRFIUNet
        A trained model (see :func:`load_trained_model`).
    dataset_dir : str
        Path to the dataset root (contains metadata.csv).
    batch_size : int
    device : torch.device, optional
    threshold : float
        Probability threshold for turning predicted probabilities into a
        binary mask.

    Returns
    -------
    dict
        Averaged "accuracy", "precision", "recall", "f1", "iou" over the
        whole test set.
    """
    if device is None:
        device = next(model.parameters()).device

    test_dataset = RFIPatchDataset(dataset_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    batch_metrics = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        batch_metrics.append(
            compute_segmentation_metrics(logits, labels, threshold=threshold)
        )

    return _average_metrics(batch_metrics)


@torch.no_grad()
def visualize_predictions(
    model: SimpleRFIUNet,
    dataset_dir: str,
    split: str = "test",
    n_samples: int = 4,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> plt.Figure:
    """
    Plot a grid of (amplitude, ground truth, prediction, overlay) rows for
    a handful of randomly chosen patches, so students can visually sanity
    check the model's predictions.

    Parameters
    ----------
    model : SimpleRFIUNet
    dataset_dir : str
    split : str
        Which split to draw examples from (usually "test").
    n_samples : int
        Number of example patches to display.
    threshold : float
        Probability threshold for the binary predicted mask.
    device : torch.device, optional
    seed : int
        Random seed controlling which patches are chosen.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if device is None:
        device = next(model.parameters()).device

    dataset = RFIPatchDataset(dataset_dir, split=split)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for row, idx in enumerate(indices):
        image_tensor, label_tensor = dataset[idx]
        image_batch = image_tensor.unsqueeze(0).to(device)  # add batch dim
        logits = model(image_batch)
        probs = torch.sigmoid(logits)
        pred_mask = (probs >= threshold).float().cpu().numpy()[0, 0]

        image_np = image_tensor.numpy()[0]
        label_np = label_tensor.numpy()[0]

        plot_waterfall(image_np, title=f"Patch {idx}: amplitude (input)", ax=axes[row, 0])
        plot_mask(label_np, title="Ground truth (AOFlagger)", ax=axes[row, 1])
        plot_mask(pred_mask, title="Predicted RFI mask", ax=axes[row, 2])
        plot_overlay(image_np, pred_mask, title="Amplitude + prediction", ax=axes[row, 3])

    fig.tight_layout()
    return fig


def print_metrics_report(metrics: Dict[str, float]) -> None:
    """Pretty-print a metrics dictionary as a small report."""
    print("Test set evaluation")
    print("--------------------")
    for name in ["accuracy", "precision", "recall", "f1", "iou"]:
        if name in metrics:
            print(f"  {name:10s}: {metrics[name]:.4f}")
