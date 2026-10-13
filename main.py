"""
main.py
=======

Command-line entry point for Part 1 of the workshop: build the
ML-ready patch dataset from a MeerKAT ``.rdb`` file (or from synthetic
data, for testing without real data).

Usage
-----
    # Using a real MeerKAT .rdb file:
    python main.py --rdb-path meerkat_data.rdb --config configs/dataset.yaml

    # Using synthetic data (no katdal / real data required):
    python main.py --synthetic --config configs/dataset.yaml
"""

import argparse

from data.dataset import DatasetConfig, create_dataset
from data.loader import load_observation, make_synthetic_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ML-ready RFI-detection dataset from MeerKAT visibility data."
    )
    parser.add_argument(
        "--rdb-path",
        type=str,
        default=None,
        help="Path to a MeerKAT .rdb file (requires katdal to be installed).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of a real .rdb file (for teaching/testing).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to the dataset YAML configuration file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output dataset directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.synthetic and args.rdb_path is None:
        raise SystemExit(
            "You must pass either --rdb-path <file.rdb> or --synthetic. "
            "Run with --help for details."
        )

    config = DatasetConfig.from_yaml(args.config)

    if args.synthetic:
        print("Generating synthetic MeerKAT-like observation for teaching purposes...")
        observation = make_synthetic_observation()
    else:
        print(f"Loading MeerKAT observation from {args.rdb_path} ...")
        observation = load_observation(args.rdb_path)

    print(
        f"Loaded observation: {observation.n_time} time steps, "
        f"{observation.n_freq} frequency channels, "
        f"{observation.n_baseline} baselines "
        f"({observation.flagged_fraction * 100:.2f}% flagged)."
    )

    print(f"Building patch dataset (patch_size={config.patch_size}, stride={config.stride}) ...")
    metadata_path = create_dataset(observation, config, overwrite=args.overwrite)

    print(f"Done! Dataset written to '{config.output_dir}/'.")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
