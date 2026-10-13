"""
patch_dataset.py
=================

Teaching notes
--------------
PyTorch expects training data wrapped in a ``torch.utils.data.Dataset``:
an object that knows how many examples it has (``__len__``) and how to
fetch one example by index (``__getitem__``). This module implements
exactly that for the dataset produced by ``meerkat_ml.data.dataset``,
reading directly from the ``metadata.csv`` file so it never needs to
re-scan the folder structure.

This is the "hand-off point" between the two halves of the workshop:
Part 1 students produce a folder of ``.npy`` files + ``metadata.csv``,
and Part 2 students only need to understand this one, small class to
start training a model on that data.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RFIPatchDataset(Dataset):
    """
    A PyTorch Dataset over one split (train/validation/test) of the
    patch dataset created in Part 1.

    Parameters
    ----------
    dataset_dir : str
        Path to the root ``dataset/`` directory (the one containing
        ``metadata.csv`` and the ``train/``, ``validation/``, ``test/``
        subfolders).
    split : str
        Which split to load: "train", "validation", or "test". Must
        match a value in the "split" column of ``metadata.csv``.
    transform : callable, optional
        Optional function applied to the image tensor after loading
        (e.g. for data augmentation). Signature: ``transform(image) -> image``.

    Notes
    -----
    Images are already normalized (Part 1 applied min-max / z-score / log
    normalization before saving them to disk), so no further scaling is
    done here. Labels are loaded as-is (0 = clean, 1 = RFI).
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        transform: Optional[callable] = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Could not find metadata.csv at {metadata_path}. "
                "Did you run the Part 1 dataset creation pipeline first?"
            )

        full_metadata = pd.read_csv(metadata_path)
        self.metadata = full_metadata[full_metadata["split"] == split].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {metadata_path}. "
                f"Available splits: {sorted(full_metadata['split'].unique())}"
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        filename = row["filename"]

        image_path = os.path.join(self.dataset_dir, self.split, "images", f"img_{filename}")
        label_path = os.path.join(self.dataset_dir, self.split, "labels", f"lbl_{filename}")

        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        # Add a channel dimension: (H, W) -> (1, H, W), which is the shape
        # PyTorch's Conv2d layers expect (channels, height, width).
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor
