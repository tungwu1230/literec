# Dataset Download Feature Design

## Overview

Add a `load_dataset()` function to lite-rec that automatically downloads, caches, and loads common recommendation system datasets. The first supported family is MovieLens (ml-100k, ml-1m, ml-10m, ml-25m).

## API

### `load_dataset(name, data_dir="./data", **kwargs) -> Dataset`

Downloads the named dataset (if not already cached), converts it to a standard CSV, and returns a fully initialized `Dataset` object.

The `data_dir` path is resolved relative to the current working directory at call time. Use an absolute path for reproducible cache location across sessions.

```python
from literec import load_dataset

dataset = load_dataset("ml-1m", min_rating=3.5, split="loo")
dataset = load_dataset("ml-1m", data_dir="./my_data", min_rating=3.5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | Dataset identifier (e.g. `"ml-1m"`) |
| `data_dir` | `str` | `"./data"` | Directory to store downloaded files (resolved relative to CWD) |
| `**kwargs` | | | Passed through to `Dataset()` (`min_rating`, `split`, `test_ratio`, etc.) |

**Returns:** A `Dataset` object ready for model training.

**Errors:**
- `ValueError` if `name` is not in the registry (message includes available names).
- Network and zip errors propagate naturally from `urllib` and `zipfile`.

### `available_datasets() -> list[str]`

Returns the list of supported dataset names in registry insertion order.

```python
from literec import available_datasets

print(available_datasets())
# ['ml-100k', 'ml-1m', 'ml-10m', 'ml-25m']
```

## File Structure After Download

Example for `ml-1m`:

```
./data/
└── ml-1m/
    ├── raw/              # Original extracted files
    │   └── ml-1m/        # Archive's internal directory
    │       ├── movies.dat
    │       ├── ratings.dat
    │       ├── users.dat
    │       └── README
    └── ratings.csv       # Converted standard CSV
```

Example for `ml-10m` (archive extracts to `ml-10M100K/`):

```
./data/
└── ml-10m/
    ├── raw/              # Original extracted files
    │   └── ml-10M100K/   # Archive's internal directory
    │       ├── movies.dat
    │       ├── ratings.dat
    │       ├── tags.dat
    │       └── README.html
    └── ratings.csv       # Converted standard CSV
```

- `raw/` preserves the original extracted archive contents (including the archive's internal directory structure).
- `ratings.csv` is the converted file with columns: `userId,movieId,rating,timestamp`.
- If `ratings.csv` already exists, download is skipped entirely.
- The raw ratings file is located at `{data_dir}/{name}/raw/{raw_file}`, where `raw_file` is the path inside the zip archive.

## Dataset Registry

A `DATASETS` dict in `downloader.py` stores metadata for each dataset:

```python
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
```

### Format Differences

| Dataset | Raw file | Separator | Approx. zip size | Notes |
|---------|----------|-----------|-------------------|-------|
| ml-100k | `ml-100k/u.data` | `\t` | ~5 MB | No header, TSV |
| ml-1m | `ml-1m/ratings.dat` | `::` | ~6 MB | No header, `::` delimited |
| ml-10m | `ml-10M100K/ratings.dat` | `::` | ~65 MB | No header, `::` delimited |
| ml-25m | `ml-25m/ratings.csv` | `,` | ~250 MB | Has header, already CSV |

All are converted to a standard CSV with header: `userId,movieId,rating,timestamp`.

For `ml-25m`, since the raw file is already a conforming CSV, use `shutil.copy` instead of a pandas round-trip to avoid loading 25M rows into memory.

## Download Flow

1. Resolve `data_dir` to an absolute path via `Path(data_dir).resolve()`.
2. Check if `{data_dir}/{name}/ratings.csv` exists → if yes, skip to step 8.
3. Download zip to a temporary file (`{data_dir}/{name}.zip`) using `urllib.request.urlretrieve`.
4. Extract zip contents to `{data_dir}/{name}/raw/` using `zipfile`.
5. Delete the temporary zip file.
6. Read the raw ratings file from `{data_dir}/{name}/raw/{raw_file}` and convert to standard CSV at `{data_dir}/{name}/ratings.csv`. For `ml-25m` (already CSV), use `shutil.copy` instead.
7. Write the CSV to a temporary file first, then rename atomically to `ratings.csv` to prevent partial files on interruption.
8. Construct and return `Dataset("{data_dir}/{name}/ratings.csv", **kwargs)`.

**Error cleanup:** Steps 3-7 are wrapped in a try/except. On failure:
- Delete the temporary zip file if it exists.
- Delete any partially-written `ratings.csv`.
- Re-raise the exception.

This prevents corrupted files from blocking future download attempts.

## Progress Display

- If `tqdm` is installed: show a download progress bar (consistent with existing training progress bar pattern).
- If `tqdm` is not installed: print `Downloading {name}...` and `Download complete.` messages.

Uses the same optional tqdm pattern already established in the project.

## Code Changes

### New file: `literec/data/downloader.py`

Uses `from __future__ import annotations` to match existing codebase convention.

Contains:
- `DATASETS` registry dict
- `load_dataset(name, data_dir, **kwargs)` function
- `available_datasets()` function
- Internal helpers for download, extraction, and conversion

### Modified file: `literec/data/__init__.py`

Add re-export:
```python
from literec.data.downloader import load_dataset, available_datasets
```

### Modified file: `literec/__init__.py`

Add exports using absolute imports (matching existing style):
```python
from literec.data import load_dataset, available_datasets
```

Add `"load_dataset"` and `"available_datasets"` to `__all__`.

### Modified file: `.gitignore`

Add `data/` to prevent downloaded datasets from being committed.

### Modified file: `pyproject.toml`

No changes needed. All functionality uses stdlib (`urllib.request`, `zipfile`, `pathlib`, `shutil`) plus existing `pandas` dependency.

## Testing

### New file: `tests/test_downloader.py`

| Test | Description |
|------|-------------|
| `test_available_datasets` | Returns correct list of names |
| `test_invalid_name_raises` | `ValueError` with helpful message for unknown names |
| `test_convert_tab_separated` | TSV (ml-100k format) converts to correct CSV |
| `test_convert_double_colon` | `::` separated (ml-1m format) converts to correct CSV |
| `test_convert_csv_with_header` | CSV with header (ml-25m format) converts correctly via copy |
| `test_cache_skip` | No download when `ratings.csv` already exists |
| `test_load_dataset_returns_dataset` | Returns `Dataset` with valid `n_users`, `n_items` |
| `test_kwargs_passthrough` | `min_rating`, `split` etc. forwarded to `Dataset()` |
| `test_bad_zip_cleans_up` | Corrupted zip is cleaned up, no partial files left |

Tests mock the download step to avoid network dependency. Format conversion tests use small synthetic data files.

## Design Decisions

1. **Independent function over class method:** `load_dataset()` is a standalone function, not `Dataset.load()`. Keeps `Dataset` focused on data processing; downloading is a separate concern.
2. **Local `./data/` over home directory:** Files are visible in the project directory, easy to inspect. Path is CWD-relative; use absolute paths for deterministic behavior.
3. **Preserve raw files:** Users can inspect original data. The `raw/` subdirectory keeps things organized. Raw file paths match the archive's internal structure.
4. **No new dependencies:** Uses stdlib for download/extraction, existing pandas for format conversion. `shutil.copy` for already-conforming CSV (ml-25m) to avoid unnecessary memory usage.
5. **Cache by file existence:** Simple check for `ratings.csv` — no version tracking or checksums. Sufficient for stable, versioned datasets like MovieLens.
6. **Hardcoded registry:** A dict is sufficient for 4 MovieLens variants. Can be refactored to a class-based approach when more dataset families are added.
7. **Atomic writes and cleanup:** Temporary file + rename for `ratings.csv` prevents partial files. Failed downloads are cleaned up automatically.
8. **Add `data/` to `.gitignore`:** Prevent accidental commit of downloaded datasets.
