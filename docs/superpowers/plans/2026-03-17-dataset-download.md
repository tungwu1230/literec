# Dataset Download Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `load_dataset()` and `available_datasets()` to lite-rec for automatic MovieLens dataset download, caching, and loading.

**Architecture:** Single new module `literec/data/downloader.py` with a dict registry mapping dataset names to download metadata. `load_dataset()` downloads a zip, extracts to `raw/`, converts to standard CSV, and returns a `Dataset` object. Cached files are reused on subsequent calls.

**Tech Stack:** Python stdlib (`urllib.request`, `zipfile`, `pathlib`, `shutil`, `os`), existing `pandas` dependency, optional `tqdm`.

**Spec:** `docs/superpowers/specs/2026-03-17-dataset-download-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `literec/data/downloader.py` | Create | `DATASETS` registry, `load_dataset()`, `available_datasets()`, internal helpers |
| `literec/data/__init__.py` | Modify | Re-export `load_dataset`, `available_datasets`; update `__all__` |
| `literec/__init__.py` | Modify | Re-export `load_dataset`, `available_datasets`; update `__all__` |
| `.gitignore` | Modify | Add `data/` |
| `tests/test_downloader.py` | Create | All tests for the download feature |

---

## Task 1: Registry and `available_datasets()`

**Files:**
- Create: `tests/test_downloader.py`
- Create: `literec/data/downloader.py`

- [ ] **Step 1: Write failing tests for `available_datasets()` and invalid name**

```python
# tests/test_downloader.py
import pytest

from literec.data.downloader import available_datasets, load_dataset


def test_available_datasets():
    result = available_datasets()
    assert result == ["ml-100k", "ml-1m", "ml-10m", "ml-25m"]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset 'not-a-dataset'"):
        load_dataset("not-a-dataset")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'literec.data.downloader'`

- [ ] **Step 3: Implement registry, `available_datasets()`, and name validation in `load_dataset()`**

```python
# literec/data/downloader.py
from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py::test_available_datasets tests/test_downloader.py::test_invalid_name_raises -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add literec/data/downloader.py tests/test_downloader.py
git commit -m "feat: add dataset registry and available_datasets()"
```

---

## Task 2: Format conversion helpers

**Files:**
- Modify: `tests/test_downloader.py`
- Modify: `literec/data/downloader.py`

- [ ] **Step 1: Write failing tests for format conversion**

Append to `tests/test_downloader.py`:

```python
from literec.data.downloader import _convert_raw_to_csv


def test_convert_tab_separated(tmp_path):
    """TSV format (ml-100k style)."""
    raw = tmp_path / "u.data"
    raw.write_text("1\t10\t5.0\t100\n2\t20\t4.0\t200\n")
    out = tmp_path / "ratings.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep="\t",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=None,
    )

    lines = out.read_text().strip().split("\n")
    assert lines[0] == "userId,movieId,rating,timestamp"
    assert lines[1] == "1,10,5.0,100"
    assert lines[2] == "2,20,4.0,200"


def test_convert_double_colon(tmp_path):
    """:: delimited format (ml-1m style)."""
    raw = tmp_path / "ratings.dat"
    raw.write_text("1::10::5.0::100\n2::20::4.0::200\n")
    out = tmp_path / "ratings.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep="::",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=None,
    )

    lines = out.read_text().strip().split("\n")
    assert lines[0] == "userId,movieId,rating,timestamp"
    assert lines[1] == "1,10,5.0,100"


def test_convert_csv_with_header(tmp_path):
    """Already-CSV format (ml-25m style) uses shutil.copy."""
    raw = tmp_path / "ratings.csv"
    raw.write_text("userId,movieId,rating,timestamp\n1,10,5.0,100\n2,20,4.0,200\n")
    out = tmp_path / "output.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep=",",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=0,
    )

    assert out.read_text() == raw.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloader.py::test_convert_tab_separated tests/test_downloader.py::test_convert_double_colon tests/test_downloader.py::test_convert_csv_with_header -v`
Expected: FAIL — `ImportError: cannot import name '_convert_raw_to_csv'`

- [ ] **Step 3: Implement `_convert_raw_to_csv`**

Add to `literec/data/downloader.py`:

```python
import os
import shutil

import pandas as pd


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py::test_convert_tab_separated tests/test_downloader.py::test_convert_double_colon tests/test_downloader.py::test_convert_csv_with_header -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add literec/data/downloader.py tests/test_downloader.py
git commit -m "feat: add format conversion helper for dataset download"
```

---

## Task 3: Download, extract, and `load_dataset()` core flow

**Files:**
- Modify: `tests/test_downloader.py`
- Modify: `literec/data/downloader.py`

- [ ] **Step 1: Write failing tests for `load_dataset()` with mocked download**

Append to `tests/test_downloader.py`:

```python
import zipfile
from pathlib import Path
from unittest.mock import patch

from literec.data.dataset import Dataset


def _create_fake_zip(zip_path, inner_file, content):
    """Helper to create a zip containing a single file."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(inner_file, content)


def _mock_urlretrieve(url, filename, reporthook=None):
    """Mock that creates a zip mimicking ml-1m format."""
    content = "1::10::5.0::100\n1::20::4.0::200\n1::30::3.0::300\n"
    content += "1::40::5.0::400\n1::50::4.0::500\n"
    content += "2::10::4.0::100\n2::20::5.0::200\n2::30::3.0::300\n"
    content += "2::40::4.0::400\n2::50::5.0::500\n"
    content += "3::10::5.0::100\n3::20::4.0::200\n3::30::5.0::300\n"
    content += "3::40::3.0::400\n3::50::4.0::500\n"
    _create_fake_zip(Path(filename), "ml-1m/ratings.dat", content)


def test_load_dataset_returns_dataset(tmp_path):
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        ds = load_dataset("ml-1m", data_dir=tmp_path)

    assert isinstance(ds, Dataset)
    assert ds.n_users == 3
    assert ds.n_items == 5


def test_kwargs_passthrough(tmp_path):
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        ds = load_dataset("ml-1m", data_dir=tmp_path, min_rating=4.5, min_interactions=3)

    # min_rating=4.5 filters most interactions, min_interactions=3 drops users
    assert ds.n_users == 0


def test_cache_skip(tmp_path):
    """Second call should not re-download."""
    call_count = 0

    def counting_urlretrieve(url, filename, reporthook=None):
        nonlocal call_count
        call_count += 1
        _mock_urlretrieve(url, filename, reporthook)

    with patch("literec.data.downloader.urllib.request.urlretrieve", counting_urlretrieve):
        load_dataset("ml-1m", data_dir=tmp_path)
        load_dataset("ml-1m", data_dir=tmp_path)

    assert call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_downloader.py::test_load_dataset_returns_dataset tests/test_downloader.py::test_kwargs_passthrough tests/test_downloader.py::test_cache_skip -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement the full `load_dataset()` flow**

Replace the placeholder `load_dataset()` in `literec/data/downloader.py` with the complete implementation:

```python
import urllib.request
import zipfile


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add literec/data/downloader.py tests/test_downloader.py
git commit -m "feat: implement load_dataset() with download, extract, convert flow"
```

---

## Task 4: Error cleanup tests

**Files:**
- Modify: `tests/test_downloader.py`

- [ ] **Step 1: Write failing tests for error cleanup**

Append to `tests/test_downloader.py`:

```python
def test_bad_zip_cleans_up(tmp_path):
    """Corrupted zip: temp files cleaned up, no partial state."""
    def bad_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"this is not a zip")

    with patch("literec.data.downloader.urllib.request.urlretrieve", bad_urlretrieve):
        with pytest.raises(zipfile.BadZipFile):
            load_dataset("ml-1m", data_dir=tmp_path)

    # No zip or temp csv left
    assert not (tmp_path / "ml-1m.zip").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv.tmp").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv").exists()


def test_conversion_failure_cleans_up(tmp_path):
    """Failed conversion: temp csv cleaned up."""
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        with patch(
            "literec.data.downloader._convert_raw_to_csv",
            side_effect=RuntimeError("conversion failed"),
        ):
            with pytest.raises(RuntimeError, match="conversion failed"):
                load_dataset("ml-1m", data_dir=tmp_path)

    assert not (tmp_path / "ml-1m.zip").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv.tmp").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv").exists()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_downloader.py::test_bad_zip_cleans_up tests/test_downloader.py::test_conversion_failure_cleans_up -v`
Expected: PASS (cleanup logic already implemented in Task 3)

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_downloader.py
git commit -m "test: add error cleanup tests for dataset download"
```

---

## Task 5: Wire up exports and `.gitignore`

**Files:**
- Modify: `literec/data/__init__.py`
- Modify: `literec/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Update `literec/data/__init__.py`**

Add the new import and update `__all__`. After editing, the file should look like:

```python
from literec.data.dataset import Dataset
from literec.data.dataloader import TrainDataLoader
from literec.data.downloader import load_dataset, available_datasets

__all__ = ["Dataset", "TrainDataLoader", "load_dataset", "available_datasets"]
```

Edits:
- Add line `from literec.data.downloader import load_dataset, available_datasets` after existing imports.
- Add `"load_dataset"` and `"available_datasets"` to `__all__`.

- [ ] **Step 2: Update `literec/__init__.py`**

Add the new imports and update `__all__`. After editing, the file should look like:

```python
from literec.config import LiteRecConfig
from literec.data import Dataset, TrainDataLoader, load_dataset, available_datasets
from literec.model import BPR, LightGCN, NGCF
from literec.training import Trainer
from literec.evaluation import Evaluator

__all__ = [
    "LiteRecConfig",
    "Dataset",
    "TrainDataLoader",
    "load_dataset",
    "available_datasets",
    "BPR",
    "LightGCN",
    "NGCF",
    "Trainer",
    "Evaluator",
]
```

Edits:
- Add `load_dataset, available_datasets` to the existing `from literec.data import ...` line.
- Add `"load_dataset"` and `"available_datasets"` to `__all__`.

- [ ] **Step 3: Add `data/` to `.gitignore`**

Append to `.gitignore`:

```
data/
```

- [ ] **Step 4: Write import test**

Append to `tests/test_downloader.py`:

```python
def test_public_import():
    """load_dataset and available_datasets accessible from top-level package."""
    from literec import load_dataset as ld, available_datasets as ad
    assert callable(ld)
    assert callable(ad)
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add literec/data/__init__.py literec/__init__.py .gitignore tests/test_downloader.py
git commit -m "feat: export load_dataset/available_datasets and add data/ to gitignore"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run full test suite one more time**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify imports work from a clean Python session**

Run: `python -c "from literec import load_dataset, available_datasets; print(available_datasets())"`
Expected: `['ml-100k', 'ml-1m', 'ml-10m', 'ml-25m']`

- [ ] **Step 3: Verify `.gitignore` includes `data/`**

Run: `grep 'data/' .gitignore`
Expected: `data/`
