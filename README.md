# MeerKAT RFI Machine Learning Workshop

A one-week educational project that teaches students how to turn raw
MeerKAT radio telescope visibility data into a machine-learning-ready
dataset (**Part 1**), and how to train a convolutional neural network
(CNN) to detect radio frequency interference (RFI) from it (**Part 2**).

This is a *teaching* codebase, not a production pipeline: the code is
deliberately kept simple, modular, and heavily commented, with an
emphasis on readability over performance or cleverness.

## Background

MeerKAT observations are stored in `.rdb` files, readable with the
[`katdal`](https://github.com/ska-sa/katdal) package. For every pair of
antennas (a *baseline*), at every integration time and frequency channel,
the correlator produces one complex number (a *visibility*). MeerKAT's
online pipeline runs an automatic flagger (AOFlagger) that marks likely
RFI-contaminated samples - we use those flags as ground-truth labels for
supervised machine learning.

```
vis.shape   == (time, frequency, baseline)   # complex64
flags.shape == (time, frequency, baseline)   # 0 = clean, 1 = RFI
```

## Project structure

```
meerkat_ml/
    data/
        loader.py           # load a .rdb file (or generate synthetic data)
        preprocessing.py    # complex visibility -> amplitude / phase / labels
        patching.py          # cut waterfalls into fixed-size patches
        normalization.py    # min-max / z-score / log normalization
        visualization.py    # waterfall / mask / overlay plotting
        dataset.py           # orchestrates the full Part 1 pipeline
    models/
        cnn.py               # a small U-Net for RFI segmentation
        patch_dataset.py     # PyTorch Dataset reading the exported patches
        train.py             # training loop
        evaluate.py           # test-set metrics + prediction visualization
    configs/
        dataset.yaml         # Part 1 configuration
        train.yaml            # Part 2 configuration
    notebooks/
        01_explore_data.ipynb    # Part 1: explore the raw data
        02_create_dataset.ipynb  # Part 1: build the patch dataset
        03_train_cnn.ipynb        # Part 2: train and evaluate the CNN
    main.py         # CLI: build the dataset (Part 1)
    train_cnn.py    # CLI: train the CNN (Part 2)
```

## Getting started

### Install dependencies

Part 1 (dataset creation) needs only:

```bash
pip install numpy pandas matplotlib pyyaml
pip install katdal   # only needed to read real .rdb files
```

Part 2 (CNN training) additionally needs:

```bash
pip install torch
```

### Part 1 - build the dataset

No real data? No problem - a synthetic-data generator is built in, so you
can run the whole pipeline immediately:

```bash
cd meerkat_ml
python main.py --synthetic --config configs/dataset.yaml --overwrite
```

With a real MeerKAT `.rdb` file:

```bash
python main.py --rdb-path meerkat_data.rdb --config configs/dataset.yaml
```

This produces:

```
dataset/
    train/images/img_000001.npy   train/labels/lbl_000001.npy
    validation/...
    test/...
    metadata.csv
```

Walk through the process interactively in `notebooks/01_explore_data.ipynb`
and `notebooks/02_create_dataset.ipynb`.

### Part 2 - train the CNN

```bash
python train_cnn.py --config configs/train.yaml
```

This trains a small U-Net segmentation model, saves the best checkpoint
(by validation F1) to `checkpoints/best_model.pt`, and reports test-set
metrics (accuracy, precision, recall, F1, IoU). Walk through it
interactively in `notebooks/03_train_cnn.ipynb`, which also visualizes
example predictions.

## Configuration

All tunable parameters live in YAML files under `configs/`, so students
can experiment (patch size, stride, normalization method, filtering
rules, dataset splits, model width, loss function, learning rate, ...)
without touching any code. See `configs/dataset.yaml` and
`configs/train.yaml` for the full list, each documented inline.

## A note on scope

This project is **not** about building a state-of-the-art RFI detector.
The goal is for students to understand, end to end, how raw
interferometric data becomes a supervised learning problem: what a
visibility is, why amplitude/phase matter, how patching and
normalization work, why class imbalance matters for RFI detection, and
how a CNN can learn to reproduce a flagging pipeline's decisions. The
CNN architecture (a 2-level U-Net) is intentionally small and simple -
extending it is a great follow-up exercise.
