"""
normalization.py
================

Teaching notes
--------------
Raw visibility amplitudes can span many orders of magnitude: quiet sky
noise might be ~1, while a strong RFI spike might be ~10,000. Neural
networks train much better when their inputs are on a consistent, small
numeric scale. This module implements three common normalization schemes
so students can experiment and see how each one changes what the network
"sees":

- **min-max**: rescale values to [0, 1] using the patch's own min/max.
  Simple and bounded, but sensitive to a single extreme outlier (like a
  very bright RFI spike) squashing everything else toward 0.
- **z-score**: subtract the mean and divide by the standard deviation.
  Centers data around 0 with unit variance; still sensitive to outliers,
  but less so than min-max.
- **log**: apply ``log(1 + x)`` before scaling. Compresses the dynamic
  range, which is often a good fit for radio astronomy data where
  amplitude is strictly positive and can vary over many orders of
  magnitude.

All functions here operate on a single 2D array (e.g. one patch) and
return a normalized 2D array of the same shape. Keeping normalization
"per patch" (rather than computed once over the whole observation) is a
deliberate teaching simplification: it keeps the code simple and avoids
having to store separate normalization statistics, at the cost of not
being the most statistically rigorous approach. This trade-off is worth
discussing with students.
"""

import numpy as np

# Small constant to avoid division by zero for perfectly flat patches.
_EPSILON = 1e-8


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    Rescale an image to the [0, 1] range using its own min and max.

    Parameters
    ----------
    image : np.ndarray
        2D array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array, same shape, dtype float32, values in [0, 1].
    """
    image = image.astype(np.float32)
    img_min = image.min()
    img_max = image.max()
    return (image - img_min) / (img_max - img_min + _EPSILON)


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """
    Standardize an image to zero mean and unit standard deviation.

    Parameters
    ----------
    image : np.ndarray
        2D array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array, same shape, dtype float32, mean ~0, std ~1.
    """
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + _EPSILON)


def normalize_log(image: np.ndarray) -> np.ndarray:
    """
    Apply a log(1 + x) compression, then rescale to [0, 1].

    Because amplitude is always non-negative, ``log1p`` is safe here
    (no need to handle negative inputs). This is the recommended default
    for amplitude waterfalls since RFI spikes can otherwise dominate a
    min-max or z-score normalization.

    Parameters
    ----------
    image : np.ndarray
        2D array to normalize. Expected to be non-negative (e.g.
        amplitude); negative values are clipped to 0 before the log.

    Returns
    -------
    np.ndarray
        Normalized array, same shape, dtype float32, values in [0, 1].
    """
    image = image.astype(np.float32)
    image = np.clip(image, a_min=0.0, a_max=None)
    log_image = np.log1p(image)
    return normalize_minmax(log_image)


# A simple registry mapping config-file strings to functions. Using a
# dictionary like this (instead of a long if/elif chain) makes it easy to
# add a new normalization method later: just write the function and add
# one line here.
NORMALIZATION_METHODS = {
    "minmax": normalize_minmax,
    "min-max": normalize_minmax,
    "zscore": normalize_zscore,
    "z-score": normalize_zscore,
    "log": normalize_log,
    "logarithmic": normalize_log,
}


def normalize(image: np.ndarray, method: str) -> np.ndarray:
    """
    Normalize an image using the named method.

    Parameters
    ----------
    image : np.ndarray
        2D array to normalize.
    method : str
        One of "minmax", "zscore", "log" (a few spelling variants are
        also accepted, see ``NORMALIZATION_METHODS``).

    Returns
    -------
    np.ndarray
        Normalized array, same shape, dtype float32.

    Raises
    ------
    ValueError
        If ``method`` is not a recognized normalization method.
    """
    key = method.lower().strip()
    if key not in NORMALIZATION_METHODS:
        valid = sorted(set(NORMALIZATION_METHODS.keys()))
        raise ValueError(
            f"Unknown normalization method '{method}'. Valid options: {valid}"
        )
    return NORMALIZATION_METHODS[key](image)
