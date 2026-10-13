"""
preprocessing.py
================

Teaching notes (radio astronomy background)
--------------------------------------------
Each visibility sample is a complex number ``V = A * e^(i*phi)``. It has:

- an **amplitude** ``|V|`` (how strong the correlated signal is), and
- a **phase** ``phi`` (related to the geometric delay between the two
  antennas for the observed sky position).

RFI usually shows up as a sudden, often very large, increase in amplitude
(and/or a scrambled phase) that doesn't follow the smooth behaviour of
astronomical signals and instrumental noise. This makes amplitude (and
sometimes phase) waterfalls a natural input for an RFI-detection model.

This module extracts simple 2D "waterfall" images - one per baseline -
from the (time, frequency, baseline) visibility cube. A waterfall image
has time on one axis and frequency on the other, which is exactly how
radio astronomers visually inspect data for RFI.
"""

from typing import Tuple

import numpy as np


def extract_amplitude(vis: np.ndarray, baseline: int) -> np.ndarray:
    """
    Extract the amplitude waterfall for a single baseline.

    Parameters
    ----------
    vis : np.ndarray
        Complex visibility cube, shape (time, frequency, baseline).
    baseline : int
        Index of the baseline to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (time, frequency) with dtype float32, equal to
        ``np.abs(vis[:, :, baseline])``.
    """
    return np.abs(vis[:, :, baseline]).astype(np.float32)


def extract_phase(vis: np.ndarray, baseline: int) -> np.ndarray:
    """
    Extract the phase waterfall (in radians) for a single baseline.

    Parameters
    ----------
    vis : np.ndarray
        Complex visibility cube, shape (time, frequency, baseline).
    baseline : int
        Index of the baseline to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (time, frequency) with dtype float32, equal to
        ``np.angle(vis[:, :, baseline])``. Values are in [-pi, pi].
    """
    return np.angle(vis[:, :, baseline]).astype(np.float32)


def extract_labels(flags: np.ndarray, baseline: int) -> np.ndarray:
    """
    Extract the ground-truth RFI flag mask for a single baseline.

    Parameters
    ----------
    flags : np.ndarray
        Flag cube, shape (time, frequency, baseline). 0 = clean, 1 = RFI.
    baseline : int
        Index of the baseline to extract.

    Returns
    -------
    np.ndarray
        2D array of shape (time, frequency) with dtype uint8.
    """
    return flags[:, :, baseline].astype(np.uint8)


def extract_baseline_waterfalls(
    vis: np.ndarray, flags: np.ndarray, baseline: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function bundling amplitude, phase, and label extraction
    for a single baseline.

    Returns
    -------
    (amplitude, phase, labels) : tuple of np.ndarray
        Each of shape (time, frequency).
    """
    amplitude = extract_amplitude(vis, baseline)
    phase = extract_phase(vis, baseline)
    labels = extract_labels(flags, baseline)
    return amplitude, phase, labels
