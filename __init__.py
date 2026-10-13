"""
meerkat_ml
==========

An educational package for turning raw MeerKAT radio telescope visibility
data into a machine-learning-ready dataset, and for training a small
convolutional neural network (CNN) to detect radio frequency interference
(RFI).

This package is intentionally written as a *teaching* codebase:
- Every module has a single, clear responsibility.
- Functions and classes are documented with docstrings that explain both
  the *code* and the underlying *radio astronomy* concepts.
- Readability is prioritized over performance or cleverness.

Package layout
--------------
meerkat_ml/
    data/           -> loading, preprocessing, patching, normalization,
                        visualization and dataset export
    models/         -> the CNN model, PyTorch dataset loader, training and
                        evaluation code (Part 2 of the workshop)
    configs/        -> YAML configuration files
    notebooks/      -> teaching notebooks
"""

__version__ = "1.0.0"
