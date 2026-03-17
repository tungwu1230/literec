from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from literec.data.dataset import Dataset

DATASETS = {
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "raw_file": "ml-100k/u.data",
        "sep": "\t",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "header": None,
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "raw_file": "ml-1m/ratings.dat",
        "sep": "::",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "header": None,
    },
    "ml-10m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "raw_file": "ml-10M100K/ratings.dat",
        "sep": "::",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "header": None,
    },
    "ml-25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "raw_file": "ml-25m/ratings.csv",
        "sep": ",",
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "header": 0,
    },
}


def _convert_raw_to_csv(
    raw_path: Path,
    csv_path: Path,
    sep: str,
    columns: list[str],
    header: int | None,
) -> None:
    """Convert a raw ratings file to standard CSV with atomic write.

    For files that are already conforming CSV (sep="," and header=0),
    uses shutil.copy to avoid loading the entire file into memory.
    """
    tmp_path = csv_path.with_suffix(".csv.tmp")
    try:
        if sep == "," and header == 0:
            shutil.copy(raw_path, tmp_path)
        else:
            df = pd.read_csv(
                raw_path,
                sep=sep,
                header=header,
                names=columns if header is None else None,
                engine="python",
            )
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def available_datasets() -> list[str]:
    """Return the list of supported dataset names."""
    return list(DATASETS.keys())


def load_dataset(name: str, data_dir: str | Path = "./data", **kwargs) -> Dataset:
    """Download a dataset and return a Dataset object.

    Args:
        name: Dataset identifier (e.g. "ml-1m").
        data_dir: Directory to store downloaded files.
        **kwargs: Passed through to Dataset().

    Returns:
        A Dataset object ready for model training.

    Raises:
        ValueError: If name is not in the registry.
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )

    # Placeholder — download logic added in later tasks
    raise NotImplementedError
