"""
loader.py
=========

Teaching notes (radio astronomy background)
--------------------------------------------
A radio interferometer like MeerKAT is made up of many antennas (dishes).
Every pair of antennas is called a *baseline*. For each baseline, at every
moment in time and at every observed radio frequency, the correlator
produces a single complex number called a *visibility*. The visibility
encodes the amplitude and phase of the correlated signal between the two
antennas, and (after calibration and imaging) is what lets us reconstruct
an image of the sky.

Radio Frequency Interference (RFI) - e.g. from satellites, phones, or
other transmitters - contaminates visibilities. MeerKAT's online pipeline
runs an automatic flagger (AOFlagger) that marks contaminated samples with
a boolean flag. In this project we treat those flags as *ground truth
labels* for a supervised machine learning problem: "given the visibility
data, predict which samples are RFI".

This module is responsible for ONE thing only: opening a MeerKAT ``.rdb``
file with ``katdal`` and handing back the raw visibility cube, the flag
cube, and a bit of useful metadata. It does not do any preprocessing,
patching, or normalization - that happens in other modules, so that each
part of the pipeline is easy to test and reason about independently.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeerKATObservation:
    """
    A simple container for everything we need from a MeerKAT observation.

    Attributes
    ----------
    vis : np.ndarray
        Complex visibilities with shape (time, frequency, baseline).
    flags : np.ndarray
        Boolean/int RFI flags with shape (time, frequency, baseline).
        0 = clean sample, 1 = RFI-contaminated sample.
    timestamps : np.ndarray
        1D array of length `time`, the timestamp (in seconds) of each
        integration.
    freqs : np.ndarray
        1D array of length `frequency`, the sky frequency (in Hz) of each
        channel.
    baseline_labels : list[str]
        Human-readable label for each baseline (e.g. "m000-m001"), length
        `baseline`.
    """

    vis: np.ndarray
    flags: np.ndarray
    timestamps: np.ndarray
    freqs: np.ndarray
    baseline_labels: list

    @property
    def n_time(self) -> int:
        """Number of time integrations in the observation."""
        return self.vis.shape[0]

    @property
    def n_freq(self) -> int:
        """Number of frequency channels in the observation."""
        return self.vis.shape[1]

    @property
    def n_baseline(self) -> int:
        """Number of baselines (antenna pairs) in the observation."""
        return self.vis.shape[2]

    @property
    def duration_seconds(self) -> float:
        """Total duration of the observation, in seconds."""
        if self.n_time < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def freq_range_hz(self) -> tuple:
        """(min_frequency_hz, max_frequency_hz) covered by the observation."""
        return float(self.freqs.min()), float(self.freqs.max())

    @property
    def flagged_fraction(self) -> float:
        """Fraction (0-1) of all (time, freq, baseline) samples flagged as RFI."""
        return float(np.mean(self.flags))


def load_observation(
    rdb_path: str,
    pol: str = "HH",
    corrprods: str = "cross",
    scans: str = "track",
) -> MeerKATObservation:
    """
    Open a MeerKAT ``.rdb`` file with katdal and load visibilities + flags.

    This function performs the ``katdal.open`` + ``data.select`` +
    ``data.vis[:]`` / ``data.flags[:]`` steps described in the project
    brief, and packages the results (plus timestamps/frequencies/baseline
    labels) into a single :class:`MeerKATObservation` object so the rest
    of the pipeline doesn't need to know anything about katdal.

    Parameters
    ----------
    rdb_path : str
        Path to the MeerKAT ``.rdb`` file.
    pol : str, default "HH"
        Polarization product to select (e.g. "HH", "VV").
    corrprods : str, default "cross"
        Correlation products to select. "cross" selects cross-correlations
        (i.e. real baselines between two different antennas), as opposed
        to "auto" (a single antenna correlated with itself).
    scans : str, default "track"
        Which scan types to select. "track" selects scans where the
        antennas were tracking a fixed sky position (as opposed to
        slewing between targets).

    Returns
    -------
    MeerKATObservation
        A dataclass bundling the visibility cube, flag cube, and metadata.

    Notes
    -----
    ``katdal`` is an optional dependency of this teaching package - it is
    only required when you actually want to read real MeerKAT data. The
    rest of the pipeline (patching, normalization, dataset export, CNN
    training) works on plain numpy arrays and has no katdal dependency,
    so students without access to katdal/real data can still run the
    pipeline on synthetic data (see ``scripts`` or the notebooks for an
    example).
    """
    try:
        import katdal
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "katdal is required to load real MeerKAT .rdb files. "
            "Install it with `pip install katdal`, or use the synthetic "
            "data generator in meerkat_ml.data.loader.make_synthetic_observation "
            "to experiment without real data."
        ) from exc

    dataset = katdal.open(rdb_path)
    dataset.select(pol=pol, corrprods=corrprods, scans=scans)

    # vis/flags shape: (time, frequency, baseline)
    vis = dataset.vis[:]
    flags = dataset.flags[:]

    timestamps = np.asarray(dataset.timestamps)
    freqs = np.asarray(dataset.freqs)

    # katdal exposes the correlation product names, which we turn into
    # readable "antA-antB" style labels for bookkeeping / plot titles.
    baseline_labels = [
        f"{a}-{b}" for a, b in getattr(dataset, "corr_products", [])[:, 0::1]
    ] if hasattr(dataset, "corr_products") else [
        f"baseline_{i}" for i in range(vis.shape[2])
    ]
    if len(baseline_labels) != vis.shape[2]:
        # Fall back to generic labels if katdal's metadata doesn't line up
        # 1:1 with the selected baseline axis (this can happen depending
        # on the katdal version / selection).
        baseline_labels = [f"baseline_{i}" for i in range(vis.shape[2])]

    return MeerKATObservation(
        vis=vis,
        flags=flags.astype(np.uint8),
        timestamps=timestamps,
        freqs=freqs,
        baseline_labels=baseline_labels,
    )


def make_synthetic_observation(
    n_time: int = 512,
    n_freq: int = 512,
    n_baseline: int = 4,
    rfi_fraction: float = 0.05,
    seed: Optional[int] = 42,
) -> MeerKATObservation:
    """
    Generate a small synthetic observation for teaching/testing purposes.

    This lets students (and this codebase's own tests) run the full
    pipeline end-to-end without needing access to a real MeerKAT ``.rdb``
    file. It creates:

    - a smooth "sky + noise" complex visibility background
    - a handful of narrow-band and broad-band RFI streaks
    - a matching binary flag cube marking exactly where the RFI is

    Parameters
    ----------
    n_time : int
        Number of time integrations to simulate.
    n_freq : int
        Number of frequency channels to simulate.
    n_baseline : int
        Number of baselines to simulate.
    rfi_fraction : float
        Approximate fraction of samples that should be marked as RFI.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    MeerKATObservation
    """
    rng = np.random.default_rng(seed)

    shape = (n_time, n_freq, n_baseline)

    # Background: low-amplitude complex Gaussian noise, as a stand-in for
    # thermal noise plus a faint, smoothly varying astronomical signal.
    real = rng.normal(loc=0.0, scale=1.0, size=shape)
    imag = rng.normal(loc=0.0, scale=1.0, size=shape)
    vis = (real + 1j * imag).astype(np.complex64)

    flags = np.zeros(shape, dtype=np.uint8)

    n_rfi_events = max(1, int(rfi_fraction * n_time * n_freq / 200))
    for _ in range(n_rfi_events):
        for b in range(n_baseline):
            # Narrow-band RFI: a specific channel contaminated for a
            # contiguous block of time (like a satellite downlink).
            f0 = rng.integers(0, n_freq)
            width = rng.integers(1, 4)
            t0 = rng.integers(0, n_time)
            t_len = rng.integers(10, n_time // 4 + 10)
            t1 = min(n_time, t0 + t_len)
            f1 = min(n_freq, f0 + width)

            boost = rng.uniform(5, 15)
            vis[t0:t1, f0:f1, b] += boost
            flags[t0:t1, f0:f1, b] = 1

    timestamps = np.arange(n_time, dtype=np.float64) * 8.0  # 8s dumps
    freqs = np.linspace(0.9e9, 1.67e9, n_freq)  # MeerKAT L-band-ish range
    baseline_labels = [f"synthetic_{i}" for i in range(n_baseline)]

    return MeerKATObservation(
        vis=vis,
        flags=flags,
        timestamps=timestamps,
        freqs=freqs,
        baseline_labels=baseline_labels,
    )
