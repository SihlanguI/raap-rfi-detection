"""
patching.py
===========

Teaching notes
--------------
A full observation waterfall can be huge (thousands of time steps by
thousands of frequency channels), which is both too large to feed into a
CNN directly and contains RFI of very different sizes and shapes.
Instead, we cut each baseline's waterfall into small, fixed-size
"patches" (e.g. 128x128 pixels). Each patch becomes one training example:

    image_patch : the amplitude (or phase) values in that patch
    label_patch : the corresponding AOFlagger RFI mask in that patch

This module implements a simple *sliding window* patch extractor:

- ``patch_size`` controls how big each patch is along the time axis
  (and, by default, the frequency axis too - patches are square unless
  you say otherwise).
- ``patch_size_freq`` optionally overrides the frequency-axis patch size
  independently of the time axis, producing rectangular patches. This
  matters a lot for real data: many MeerKAT scans have very few time
  dumps (e.g. a short calibration "track" with only 20-30 integrations)
  but thousands of frequency channels. Forcing a *square* patch in that
  case would either be impossibly large for the time axis, or so small
  that it throws away almost all of the frequency resolution. A
  rectangular patch (e.g. 16 time samples x 256 frequency channels) is
  often a much better fit.
- ``stride`` / ``stride_freq`` control how far the window moves between
  patches along each axis. A stride smaller than the patch size produces
  overlapping patches (useful for data augmentation / more training
  examples); a stride equal to the patch size produces non-overlapping
  patches.
- Incomplete patches at the edges of the waterfall (where a full-size
  window doesn't fit) are discarded, to keep every patch the same shape.

Note for Part 2 (CNN training): ``models.cnn.SimpleRFIUNet`` uses two
2x2 max-pooling stages, so it expects both patch dimensions to be
divisible by 4. Keep this in mind when choosing ``patch_size`` /
``patch_size_freq`` for asymmetric patches (e.g. 16, 20, 64, 128, 256
are all divisible by 4; 22 is not).
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class Patch:
    """
    A single extracted patch, with everything needed to reconstruct where
    it came from in the original waterfall (this is what eventually goes
    into ``metadata.csv``).

    Attributes
    ----------
    image : np.ndarray
        2D array of shape (patch_height, patch_width), the input data.
    label : np.ndarray
        2D array of shape (patch_height, patch_width), the RFI flag mask.
    start_time : int
        Index of the first time sample included in this patch.
    end_time : int
        Index one past the last time sample included in this patch
        (i.e. the patch covers ``[start_time, end_time)``).
    start_freq : int
        Index of the first frequency channel included in this patch.
    end_freq : int
        Index one past the last frequency channel included in this patch.
    """

    image: np.ndarray
    label: np.ndarray
    start_time: int
    end_time: int
    start_freq: int
    end_freq: int

    @property
    def flagged_fraction(self) -> float:
        """Fraction (0-1) of pixels in this patch's label that are RFI."""
        return float(np.mean(self.label))


def resolve_patch_dims(
    patch_size: int,
    stride: int,
    patch_size_freq: Optional[int] = None,
    stride_freq: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    Resolve the (possibly asymmetric) patch/stride settings used
    throughout this module into an explicit
    ``(patch_time, patch_freq, stride_time, stride_freq)`` tuple.

    ``patch_size`` and ``stride`` always apply to the time axis.
    ``patch_size_freq`` / ``stride_freq`` apply to the frequency axis; if
    left as ``None``, they default to the time-axis values (i.e. a
    square patch), which keeps this function fully backward-compatible
    with code that only ever set ``patch_size`` / ``stride``.

    Raises
    ------
    ValueError
        If any resolved value is not a positive integer.
    """
    patch_time = patch_size
    patch_freq = patch_size_freq if patch_size_freq is not None else patch_size
    stride_time = stride
    stride_f = stride_freq if stride_freq is not None else stride

    if patch_time <= 0 or patch_freq <= 0 or stride_time <= 0 or stride_f <= 0:
        raise ValueError(
            "patch_size, stride, patch_size_freq, and stride_freq must all be "
            "positive integers."
        )
    return patch_time, patch_freq, stride_time, stride_f


def extract_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: int = 128,
    stride: int = 128,
    patch_size_freq: Optional[int] = None,
    stride_freq: Optional[int] = None,
) -> Iterator[Patch]:
    """
    Slide a fixed-size window over a 2D (time, frequency) image and its
    matching label mask, yielding one :class:`Patch` per window position.

    Incomplete patches at the bottom/right edges (where a full-size
    window would run off the array) are discarded, so every yielded
    patch has exactly shape (patch_time, patch_freq).

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (time, frequency), e.g. an amplitude waterfall.
    label : np.ndarray
        2D array of shape (time, frequency), the matching RFI flag mask.
        Must have the same shape as ``image``.
    patch_size : int, default 128
        Size (in pixels) of each patch along the time axis. Also used
        for the frequency axis unless ``patch_size_freq`` is given.
    stride : int, default 128
        Step size (in pixels) between consecutive patches along the time
        axis. Also used for the frequency axis unless ``stride_freq`` is
        given. Use a stride smaller than the patch size for overlapping
        patches, or equal to it for non-overlapping (tiled) patches.
    patch_size_freq : int, optional
        Size (in pixels) of each patch along the frequency axis, if
        different from ``patch_size``. Use this to build rectangular
        patches - e.g. for a short observation with few time dumps but
        many frequency channels, a small ``patch_size`` (time) combined
        with a larger ``patch_size_freq`` makes much better use of the
        available bandwidth than forcing a square patch would.
    stride_freq : int, optional
        Step size (in pixels) between consecutive patches along the
        frequency axis, if different from ``stride``.

    Yields
    ------
    Patch
        One patch for every valid window position, in row-major
        (time first, then frequency) order.

    Raises
    ------
    ValueError
        If ``image`` and ``label`` shapes don't match, if any of the
        patch/stride settings are not positive integers, or if the
        (possibly asymmetric) patch size is larger than the image along
        the corresponding axis (in which case no patch could ever be
        extracted, and this raises instead of silently yielding nothing
        - see notes below).

    Notes
    -----
    A very common source of confusion: if the patch size is larger than
    the image's time or frequency axis, the sliding window never fits
    even once, and this function would otherwise silently yield *zero*
    patches with no error at all - which is easy to misread as "there's
    no RFI here" when actually no patches were ever extracted. Real
    MeerKAT scans (especially short calibration tracks) can easily have
    far fewer time dumps than a typical ``patch_size`` like 128, so this
    is checked explicitly and raises a clear error instead. If your data
    is very asymmetric (few time samples, many frequency channels, or
    vice versa), use ``patch_size_freq`` / ``stride_freq`` to decouple
    the two axes instead of forcing a square patch.
    """
    if image.shape != label.shape:
        raise ValueError(
            f"image shape {image.shape} does not match label shape {label.shape}"
        )

    patch_time, patch_freq, stride_time, stride_f = resolve_patch_dims(
        patch_size, stride, patch_size_freq, stride_freq
    )

    n_time, n_freq = image.shape
    if patch_time > n_time or patch_freq > n_freq:
        raise ValueError(
            f"Requested patch shape (time={patch_time}, freq={patch_freq}) does "
            f"not fit inside this image's shape (n_time={n_time}, n_freq={n_freq}), "
            f"so no patches can be extracted at all.\n"
            f"  - time axis:  patch_size={patch_time} vs. n_time={n_time} "
            f"({'OK' if patch_time <= n_time else 'TOO LARGE'})\n"
            f"  - freq axis:  patch_size_freq={patch_freq} vs. n_freq={n_freq} "
            f"({'OK' if patch_freq <= n_freq else 'TOO LARGE'})\n"
            f"If your data is asymmetric (e.g. few time samples but many "
            f"frequency channels, or vice versa), set patch_size and "
            f"patch_size_freq independently instead of using a single square "
            f"patch_size - e.g. patch_size={min(patch_time, n_time)}, "
            f"patch_size_freq={min(patch_freq, n_freq)} would fit this data."
        )

    for start_time in range(0, n_time - patch_time + 1, stride_time):
        end_time = start_time + patch_time
        for start_freq in range(0, n_freq - patch_freq + 1, stride_f):
            end_freq = start_freq + patch_freq

            image_patch = image[start_time:end_time, start_freq:end_freq]
            label_patch = label[start_time:end_time, start_freq:end_freq]

            yield Patch(
                image=image_patch,
                label=label_patch,
                start_time=start_time,
                end_time=end_time,
                start_freq=start_freq,
                end_freq=end_freq,
            )


def count_patches(
    n_time: int,
    n_freq: int,
    patch_size: int,
    stride: int,
    patch_size_freq: Optional[int] = None,
    stride_freq: Optional[int] = None,
) -> int:
    """
    Compute how many patches ``extract_patches`` would yield for a
    waterfall of shape (n_time, n_freq), without actually extracting them.

    Useful for students to sanity-check dataset size before running the
    full (potentially slow) extraction over every baseline. Accepts the
    same ``patch_size_freq`` / ``stride_freq`` overrides as
    :func:`extract_patches`, for asymmetric (rectangular) patches.
    """
    patch_time, patch_freq, stride_time, stride_f = resolve_patch_dims(
        patch_size, stride, patch_size_freq, stride_freq
    )
    n_along_time = max(0, (n_time - patch_time) // stride_time + 1)
    n_along_freq = max(0, (n_freq - patch_freq) // stride_f + 1)
    return n_along_time * n_along_freq

