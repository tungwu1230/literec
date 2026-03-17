from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
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


def _download_and_extract(url: str, zip_path: Path, raw_dir: Path) -> None:
    """Download a zip and extract to raw_dir."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm

        pbar = None

        def reporthook(count, block_size, total_size):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                )
            pbar.update(block_size)

        urllib.request.urlretrieve(url, str(zip_path), reporthook=reporthook)
        if pbar is not None:
            pbar.close()
    except ImportError:
        print(f"Downloading {zip_path.stem}...")
        urllib.request.urlretrieve(url, str(zip_path))
        print("Download complete.")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


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

    info = DATASETS[name]
    data_dir = Path(data_dir).resolve()
    dataset_dir = data_dir / name
    csv_path = dataset_dir / "ratings.csv"

    if not csv_path.exists():
        zip_path = data_dir / f"{name}.zip"
        raw_dir = dataset_dir / "raw"
        try:
            _download_and_extract(info["url"], zip_path, raw_dir)
            raw_file = raw_dir / info["raw_file"]
            _convert_raw_to_csv(
                raw_path=raw_file,
                csv_path=csv_path,
                sep=info["sep"],
                columns=info["columns"],
                header=info["header"],
            )
        except Exception:
            zip_path.unlink(missing_ok=True)
            csv_path.with_suffix(".csv.tmp").unlink(missing_ok=True)
            raise
        finally:
            zip_path.unlink(missing_ok=True)

    return Dataset(str(csv_path), **kwargs)
