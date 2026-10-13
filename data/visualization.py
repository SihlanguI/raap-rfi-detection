"""
visualization.py
=================

Teaching notes
--------------
Visual inspection is a fundamental technique in radio astronomy. A
"waterfall plot" shows time on one axis and frequency on the other, with
color representing amplitude, phase, or a binary flag. RFI often shows up
as a bright vertical stripe (narrow-band, i.e. contaminating a small
range of frequencies for a long time) or a bright horizontal stripe
(broadband, i.e. contaminating many frequencies briefly).

This module centralizes all plotting so that both the exploration
notebook and the training/evaluation notebooks use the exact same,
consistent visual style.

Convention used throughout this module (matches the project brief):
    x-axis = frequency
    y-axis = time
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_waterfall(
    image: np.ndarray,
    title: str = "Waterfall",
    ylabel: str = "Time index",
    xlabel: str = "Frequency index",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a single 2D waterfall image (e.g. amplitude or phase).

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (time, frequency).
    title : str
        Plot title.
    ylabel, xlabel : str
        Axis labels. Defaults match the project convention: time on the
        y-axis, frequency on the x-axis.
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the image was plotted on (useful for further tweaks).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # origin="lower" keeps time increasing upward, which is the usual
    # astronomical convention for waterfall plots.
    im = ax.imshow(image, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    return ax


def plot_mask(
    mask: np.ndarray,
    title: str = "AOFlagger RFI mask",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a binary RFI flag mask using a simple two-color colormap
    (clean = dark, RFI = bright).

    Parameters
    ----------
    mask : np.ndarray
        2D binary array of shape (time, frequency), 0 = clean, 1 = RFI.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(mask, aspect="auto", origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Time index")
    return ax


def plot_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "Amplitude with RFI overlay",
    image_cmap: str = "viridis",
    mask_color: str = "red",
    mask_alpha: float = 0.45,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Overlay an RFI mask on top of a waterfall image, so students can see
    exactly which pixels the flagger considered contaminated.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (time, frequency), e.g. amplitude.
    mask : np.ndarray
        2D binary array, same shape as ``image``.
    title : str
        Plot title.
    image_cmap : str
        Colormap for the background image.
    mask_color : str
        Color used to highlight flagged (RFI) pixels.
    mask_alpha : float
        Opacity of the mask overlay (0 = invisible, 1 = opaque).
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(image, aspect="auto", origin="lower", cmap=image_cmap)

    # Build an RGBA overlay that is fully transparent where mask == 0,
    # and colored + semi-transparent where mask == 1.
    from matplotlib.colors import to_rgba

    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    r, g, b, _ = to_rgba(mask_color)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = mask.astype(np.float32) * mask_alpha

    ax.imshow(rgba, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Time index")
    return ax


def plot_baseline_summary(
    amplitude: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
    baseline_label: str = "",
) -> plt.Figure:
    """
    Convenience function that reproduces the three-panel figure requested
    in the exploration notebook: amplitude waterfall, phase waterfall, and
    AOFlagger mask, side by side.

    Parameters
    ----------
    amplitude, phase, mask : np.ndarray
        2D arrays of shape (time, frequency).
    baseline_label : str
        Label identifying the baseline, used in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_waterfall(amplitude, title="Amplitude", ax=axes[0])
    plot_waterfall(phase, title="Phase", cmap="twilight", ax=axes[1])
    plot_mask(mask, title="AOFlagger mask", ax=axes[2])
    fig.suptitle(f"Baseline: {baseline_label}" if baseline_label else "")
    fig.tight_layout()
    return fig
