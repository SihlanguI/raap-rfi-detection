"""
dataset.py
==========

This is the "orchestration" module for Part 2 of the workshop: it wires
together loading -> preprocessing -> patching -> filtering ->
normalization -> train/val/test splitting -> export to disk, driven by a
single YAML configuration file.

Teaching notes
--------------
Splitting the pipeline into small, independent modules (loader,
preprocessing, patching, normalization, visualization) and then having
one place (this file) that combines them is a common and useful software
design pattern. It means:

- Each small module can be understood, tested, and modified on its own.
- This module reads almost like a checklist of the steps in the pipeline,
  which makes the overall process easy to follow.
- The next workshop group (training the CNN) only needs to understand the
  *output* of this module (the dataset/ folder + metadata.csv), not its
  internals.
"""

import csv
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from .loader import MeerKATObservation
from .normalization import normalize
from .patching import Patch, extract_patches, resolve_patch_dims
from .preprocessing import extract_amplitude, extract_labels


@dataclass
class DatasetConfig:
    """
    All the configurable parameters for dataset creation, normally loaded
    from a YAML file (see ``configs/dataset.yaml`` for an example).

    Attributes
    ----------
    patch_size : int
        Size (in pixels) of each patch along the time axis. Also used
        for the frequency axis unless ``patch_size_freq`` is set.
    stride : int
        Step size (in pixels) between consecutive patches along the time
        axis. Also used for the frequency axis unless ``stride_freq`` is
        set.
    patch_size_freq : int, optional
        Size (in pixels) of each patch along the frequency axis, if you
        want rectangular (non-square) patches. Leave unset for square
        patches. This matters for real data with very asymmetric shapes
        - e.g. a short calibration scan with only ~20 time dumps but
        thousands of frequency channels benefits from a small time-axis
        patch size combined with a much larger frequency-axis one,
        rather than forcing a square patch that either doesn't fit the
        time axis or wastes most of the available bandwidth.
    stride_freq : int, optional
        Step size (in pixels) between consecutive patches along the
        frequency axis, if different from ``stride``.
    normalization : str
        Normalization method: "minmax", "zscore", or "log".
    train_split, validation_split, test_split : float
        Fractions of the dataset assigned to each split. Must sum to 1.0.
    discard_empty : bool
        If True, discard patches that contain no RFI at all.
    discard_full_rfi : bool
        If True, discard patches that are 100% RFI.
    min_rfi_fraction : float
        Minimum fraction (0-1) of RFI pixels required to keep a patch.
    max_rfi_fraction : float
        Maximum fraction (0-1) of RFI pixels allowed to keep a patch.
    output_dir : str
        Directory the dataset will be written to.
    random_seed : int
        Seed used to shuffle patches before splitting, for reproducibility.
    """

    patch_size: int = 128
    stride: int = 64
    patch_size_freq: Optional[int] = None
    stride_freq: Optional[int] = None
    normalization: str = "zscore"
    train_split: float = 0.70
    validation_split: float = 0.15
    test_split: float = 0.15
    discard_empty: bool = False
    discard_full_rfi: bool = False
    min_rfi_fraction: float = 0.0
    max_rfi_fraction: float = 1.0
    output_dir: str = "dataset"
    random_seed: int = 42

    def resolved_patch_dims(self):
        """Return (patch_time, patch_freq, stride_time, stride_freq_)."""
        return resolve_patch_dims(
            self.patch_size, self.stride, self.patch_size_freq, self.stride_freq
        )

    def validate(self) -> None:
        """Sanity-check the configuration, raising ValueError on problems."""
        splits = [self.train_split, self.validation_split, self.test_split]
        if any(s < 0 for s in splits):
            raise ValueError("Split fractions must be non-negative.")
        total = sum(splits)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"train/validation/test splits must sum to 1.0, got {total}")
        # This also validates patch_size / stride / patch_size_freq / stride_freq
        # are all positive integers.
        self.resolved_patch_dims()
        if not (0.0 <= self.min_rfi_fraction <= self.max_rfi_fraction <= 1.0):
            raise ValueError(
                "Require 0 <= min_rfi_fraction <= max_rfi_fraction <= 1."
            )

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        """Load a :class:`DatasetConfig` from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        config = cls(**raw)
        config.validate()
        return config


@dataclass
class LabeledPatch:
    """A patch together with the bookkeeping needed to write it to disk."""

    patch: Patch
    baseline_index: int


@dataclass
class PatchMetadataRow:
    """One row of the eventual ``metadata.csv`` file."""

    filename: str
    split: str
    baseline_index: int
    start_time: int
    end_time: int
    start_freq: int
    end_freq: int
    flagged_fraction: float


def _passes_filters(patch: Patch, config: DatasetConfig) -> bool:
    """
    Decide whether a patch should be kept, based on the filtering options
    in the config. Centralizing this logic in one function (rather than
    scattering `if` statements across the pipeline) makes it easy to see
    every filtering rule at a glance.
    """
    frac = patch.flagged_fraction

    if config.discard_empty and frac == 0.0:
        return False
    if config.discard_full_rfi and frac == 1.0:
        return False
    if frac < config.min_rfi_fraction:
        return False
    if frac > config.max_rfi_fraction:
        return False
    return True


def build_patches(
    observation: MeerKATObservation,
    config: DatasetConfig,
    apply_filters: bool = True,
    verbose: bool = True,
) -> List[LabeledPatch]:
    """
    Run preprocessing + patch extraction + filtering + normalization over
    every baseline in an observation.

    Parameters
    ----------
    observation : MeerKATObservation
        The loaded observation (see ``loader.load_observation`` or
        ``loader.make_synthetic_observation``).
    config : DatasetConfig
        Dataset-creation configuration.
    apply_filters : bool, default True
        If False, skip the ``discard_empty`` / ``discard_full_rfi`` /
        ``min_rfi_fraction`` / ``max_rfi_fraction`` checks and keep every
        extracted patch. This is useful for *diagnosing* a dataset before
        deciding on filter thresholds - see
        :func:`inspect_rfi_fraction_distribution`, which uses this flag.
    verbose : bool, default True
        If True, and filtering is enabled, print a short breakdown of how
        many patches were extracted vs. discarded, and why. Real MeerKAT
        data is often much more sparsely flagged than the synthetic data
        used for testing this pipeline, so this breakdown is important
        for catching an over-strict configuration (e.g. every patch
        being discarded because ``min_rfi_fraction`` is set higher than
        the RFI density actually present in the observation).

    Returns
    -------
    list[LabeledPatch]
        All patches that passed the configured filters (or all patches,
        if ``apply_filters=False``), with their ``image`` already
        normalized according to ``config.normalization``.

    Raises
    ------
    ValueError
        If the (possibly asymmetric) configured patch size is larger
        than the observation's time or frequency axis, so that no patch
        could ever be extracted from any baseline. This is checked up
        front (before looping over baselines) so the error message can
        report the actual observation dimensions alongside the
        offending config values.
    """
    patch_time, patch_freq, _, _ = config.resolved_patch_dims()
    if patch_time > observation.n_time or patch_freq > observation.n_freq:
        raise ValueError(
            f"Configured patch shape (time={patch_time}, freq={patch_freq}) does "
            f"not fit this observation's dimensions (n_time={observation.n_time}, "
            f"n_freq={observation.n_freq}). No patches can be extracted from ANY "
            f"baseline as a result - this is a very common cause of '0 patches' "
            f"with real data, since real scans (especially short calibration "
            f"tracks) often have far fewer time dumps than a typical patch_size "
            f"like 128.\n"
            f"  - time axis:  patch_size={patch_time} vs. n_time={observation.n_time} "
            f"({'OK' if patch_time <= observation.n_time else 'TOO LARGE'})\n"
            f"  - freq axis:  patch_size_freq={patch_freq} vs. n_freq={observation.n_freq} "
            f"({'OK' if patch_freq <= observation.n_freq else 'TOO LARGE'})\n"
            f"Fix: in configs/dataset.yaml, lower patch_size (time axis) to at "
            f"most {observation.n_time}, and/or set patch_size_freq (frequency "
            f"axis, independent of patch_size) to at most {observation.n_freq}. "
            f"If your data is very asymmetric (as here), using different values "
            f"for patch_size and patch_size_freq - rather than one square "
            f"patch_size - is usually the better choice."
        )

    all_patches: List[LabeledPatch] = []

    n_extracted = 0
    n_discarded_empty = 0
    n_discarded_full = 0
    n_discarded_min_frac = 0
    n_discarded_max_frac = 0

    for baseline in range(observation.n_baseline):
        amplitude = extract_amplitude(observation.vis, baseline)
        labels = extract_labels(observation.flags, baseline)

        for patch in extract_patches(
            amplitude,
            labels,
            patch_size=config.patch_size,
            stride=config.stride,
            patch_size_freq=config.patch_size_freq,
            stride_freq=config.stride_freq,
        ):
            n_extracted += 1
            frac = patch.flagged_fraction

            if apply_filters:
                if config.discard_empty and frac == 0.0:
                    n_discarded_empty += 1
                    continue
                if config.discard_full_rfi and frac == 1.0:
                    n_discarded_full += 1
                    continue
                if frac < config.min_rfi_fraction:
                    n_discarded_min_frac += 1
                    continue
                if frac > config.max_rfi_fraction:
                    n_discarded_max_frac += 1
                    continue

            # Normalize AFTER filtering: filtering decisions are based on
            # the *label* (RFI fraction), so normalization order doesn't
            # affect which patches are kept - but doing it after filtering
            # avoids wasted work normalizing patches we're about to throw
            # away.
            normalized_image = normalize(patch.image, config.normalization)
            normalized_patch = Patch(
                image=normalized_image,
                label=patch.label,
                start_time=patch.start_time,
                end_time=patch.end_time,
                start_freq=patch.start_freq,
                end_freq=patch.end_freq,
            )
            all_patches.append(LabeledPatch(patch=normalized_patch, baseline_index=baseline))

    if apply_filters and verbose:
        n_kept = len(all_patches)
        print(
            f"Extracted {n_extracted} raw patches -> kept {n_kept} after filtering."
        )
        if n_extracted > 0 and n_kept < n_extracted:
            print(
                f"  discarded (0% RFI, discard_empty):       {n_discarded_empty}\n"
                f"  discarded (100% RFI, discard_full_rfi):  {n_discarded_full}\n"
                f"  discarded (below min_rfi_fraction={config.min_rfi_fraction}): {n_discarded_min_frac}\n"
                f"  discarded (above max_rfi_fraction={config.max_rfi_fraction}): {n_discarded_max_frac}"
            )
        if n_extracted > 0 and n_kept == 0:
            print(
                "\nWARNING: every patch was discarded! Real observations are often "
                "much more sparsely flagged than the synthetic test data, so "
                "min_rfi_fraction (or discard_empty) may be too strict for this "
                "observation. Try calling "
                "inspect_rfi_fraction_distribution(observation, config) first "
                "to see the actual per-patch RFI fractions before choosing "
                "filter thresholds."
            )
        if patch_time % 4 != 0 or patch_freq % 4 != 0:
            print(
                f"\nNOTE: patch shape (time={patch_time}, freq={patch_freq}) is not "
                f"divisible by 4 on both axes. This is fine for Part 1 (dataset "
                f"creation), but models.cnn.SimpleRFIUNet (Part 2) uses two 2x2 "
                f"max-pooling stages and requires both dimensions divisible by 4 "
                f"- adjust patch_size / patch_size_freq before training the CNN "
                f"on this dataset if you plan to use that model as-is."
            )

    return all_patches


def inspect_rfi_fraction_distribution(
    observation: MeerKATObservation, config: DatasetConfig
) -> np.ndarray:
    """
    Extract every patch with NO filtering applied, and return the array of
    per-patch flagged fractions. This is meant to be called BEFORE
    deciding on ``discard_empty`` / ``min_rfi_fraction`` / etc., so you can
    see what range of RFI density your patches actually contain.

    Real MeerKAT observations (especially after the online AOFlagger has
    already run) are often much more sparsely and unevenly flagged than
    the small synthetic examples used to test this pipeline - it is easy
    to pick a ``min_rfi_fraction`` that silently discards every single
    patch. Checking the real distribution first avoids that trap.

    Parameters
    ----------
    observation : MeerKATObservation
    config : DatasetConfig
        Only ``patch_size`` and ``stride`` are used from this config;
        all filtering settings are ignored.

    Returns
    -------
    np.ndarray
        1D array of flagged fractions (0-1), one entry per extracted
        patch, across every baseline.

    Example
    -------
    >>> fractions = inspect_rfi_fraction_distribution(observation, config)
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(fractions, bins=50)
    >>> print("percent of patches with any RFI:", (fractions > 0).mean() * 100)
    >>> print("95th percentile flagged fraction:", np.percentile(fractions, 95))
    """
    unfiltered_patches = build_patches(observation, config, apply_filters=False, verbose=False)
    return np.array([lp.patch.flagged_fraction for lp in unfiltered_patches])


def split_patches(
    patches: List[LabeledPatch], config: DatasetConfig
) -> dict:
    """
    Shuffle and split a list of patches into train/validation/test sets.

    Parameters
    ----------
    patches : list[LabeledPatch]
    config : DatasetConfig

    Returns
    -------
    dict
        Dictionary with keys "train", "validation", "test", each mapping
        to a list of LabeledPatch.
    """
    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(len(patches))
    rng.shuffle(indices)

    n_total = len(patches)
    n_train = int(round(config.train_split * n_total))
    n_val = int(round(config.validation_split * n_total))
    # Remainder goes to test, so rounding never loses/gains a patch.
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    return {
        "train": [patches[i] for i in train_idx],
        "validation": [patches[i] for i in val_idx],
        "test": [patches[i] for i in test_idx],
    }


def _make_output_dirs(output_dir: str, splits: List[str]) -> None:
    """Create the dataset/{split}/{images,labels}/ directory tree."""
    for split in splits:
        for subdir in ("images", "labels"):
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)


def export_dataset(
    split_dict: dict, config: DatasetConfig, overwrite: bool = False
) -> str:
    """
    Write patches to disk in the layout described in the project brief:

        dataset/
            train/
                images/img_000001.npy
                labels/lbl_000001.npy
            validation/
            test/
            metadata.csv

    Parameters
    ----------
    split_dict : dict
        Output of :func:`split_patches`.
    config : DatasetConfig
    overwrite : bool
        If True, delete any existing ``config.output_dir`` before writing.

    Returns
    -------
    str
        Path to the written ``metadata.csv`` file.
    """
    output_dir = config.output_dir

    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    _make_output_dirs(output_dir, list(split_dict.keys()))

    metadata_rows: List[PatchMetadataRow] = []

    # A single running counter gives every patch a unique, sortable
    # filename across the whole dataset (not just within its split).
    counter = 1

    for split_name, labeled_patches in split_dict.items():
        for labeled in labeled_patches:
            patch = labeled.patch
            filename = f"{counter:06d}.npy"
            counter += 1

            image_path = os.path.join(output_dir, split_name, "images", f"img_{filename}")
            label_path = os.path.join(output_dir, split_name, "labels", f"lbl_{filename}")

            np.save(image_path, patch.image.astype(np.float32))
            np.save(label_path, patch.label.astype(np.uint8))

            metadata_rows.append(
                PatchMetadataRow(
                    filename=filename,
                    split=split_name,
                    baseline_index=labeled.baseline_index,
                    start_time=patch.start_time,
                    end_time=patch.end_time,
                    start_freq=patch.start_freq,
                    end_freq=patch.end_freq,
                    flagged_fraction=patch.flagged_fraction,
                )
            )

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "split",
                "baseline_index",
                "start_time",
                "end_time",
                "start_freq",
                "end_freq",
                "flagged_fraction",
            ]
        )
        for row in metadata_rows:
            writer.writerow(
                [
                    row.filename,
                    row.split,
                    row.baseline_index,
                    row.start_time,
                    row.end_time,
                    row.start_freq,
                    row.end_freq,
                    f"{row.flagged_fraction:.6f}",
                ]
            )

    return metadata_path


def create_dataset(
    observation: MeerKATObservation, config: DatasetConfig, overwrite: bool = False
) -> str:
    """
    High-level convenience function: run the full pipeline
    (patch extraction -> filtering -> normalization -> splitting ->
    export) for one observation.

    Parameters
    ----------
    observation : MeerKATObservation
    config : DatasetConfig
    overwrite : bool
        If True, overwrite any existing dataset at ``config.output_dir``.

    Returns
    -------
    str
        Path to the written metadata.csv file.
    """
    config.validate()
    patches = build_patches(observation, config)
    if len(patches) == 0:
        raise RuntimeError(
            "No patches survived filtering - try relaxing discard_empty / "
            "discard_full_rfi / min_rfi_fraction / max_rfi_fraction in the config."
        )
    split_dict = split_patches(patches, config)
    return export_dataset(split_dict, config, overwrite=overwrite)
