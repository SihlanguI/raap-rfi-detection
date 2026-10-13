"""
meerkat_ml.models
==================

Part 2 of the workshop: train a convolutional neural network (CNN) to
predict RFI masks from the amplitude patches produced by
``meerkat_ml.data``.

    cnn.py            -> the CNN architecture (a small U-Net style
                          segmentation network)
    patch_dataset.py  -> a PyTorch Dataset that reads the .npy patches
                          and metadata.csv written by Part 1
    train.py          -> the training loop, driven by a YAML config
    evaluate.py        -> compute metrics and visualize predictions on
                          held-out data

These modules depend on PyTorch (``pip install torch``), which is kept
separate from the Part-1 dependencies (numpy/matplotlib/pyyaml/katdal) so
that students who only want to build the dataset don't need to install
a deep learning framework.
"""
