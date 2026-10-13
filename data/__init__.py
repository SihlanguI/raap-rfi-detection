"""
meerkat_ml.data
================

Everything needed to go from a raw MeerKAT ``.rdb`` file to a folder of
machine-learning-ready ``.npy`` patches:

    loader.py          -> read the .rdb file with katdal
    preprocessing.py   -> turn complex visibilities into amplitude/phase
    normalization.py   -> min-max / z-score / log normalization
    patching.py         -> cut waterfall images into fixed-size patches
    visualization.py   -> plotting helpers (waterfalls, masks, overlays)
    dataset.py          -> orchestrates the full pipeline and writes the
                            dataset to disk (images/, labels/, metadata.csv)
"""
