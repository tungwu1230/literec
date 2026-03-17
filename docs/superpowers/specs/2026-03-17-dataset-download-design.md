# Dataset Download Feature Design

## Overview

Add a `load_dataset()` function to lite-rec that automatically downloads, caches, and loads common recommendation system datasets. The first supported family is MovieLens (ml-100k, ml-1m, ml-10m, ml-25m).

## API

### `load_dataset(name, data_dir="./data", **kwargs) -> Dataset`

Downloads the named dataset (if not already cached), converts it to a standard CSV, and returns a fully initialized `Dataset` object.

```python
from literec import load_dataset

dataset = load_dataset("ml-1m", min_rating=3.5, split="loo")
dataset = load_dataset("ml-1m", data_dir="./my_data", min_rating=3.5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | Dataset identifier (e.g. `"ml-1m"`) |
| `data_dir` | `str` | `"./data"` | Directory to store downloaded files |
| `**kwargs` | | | Passed through to `Dataset()` (`min_rating`, `split`, `test_ratio`, etc.) |

**Returns:** A `Dataset` object ready for model training.

**Errors:**
- `ValueError` if `name` is not in the registry (message includes available names).
- Network and zip errors propagate naturally from `urllib` and `zipfile`.

### `available_datasets() -> list[str]`

Returns a sorted list of supported dataset names.

```python
from literec import available_datasets

print(available_datasets())
# ['ml-100k', 'ml-1m', 'ml-10m', 'ml-25m']
```

## File Structure After Download

```
./data/
└── ml-1m/
    ├── raw/              # Original extracted files
    │   ├── movies.dat
    │   ├── ratings.dat
    │   ├── users.dat
    │   └── README
    └── ratings.csv       # Converted standard CSV
```

- `raw/` preserves the original extracted archive contents.
- `ratings.csv` is the converted file with columns: `userId,movieId,rating,timestamp`.
- If `ratings.csv` already exists, download is skipped entirely.

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

| Dataset | Raw file | Separator | Notes |
|---------|----------|-----------|-------|
| ml-100k | `u.data` | `\t` | No header, TSV |
| ml-1m | `ratings.dat` | `::` | No header, `::` delimited |
| ml-10m | `ratings.dat` | `::` | No header, `::` delimited |
| ml-25m | `ratings.csv` | `,` | Has header, already CSV |

All are converted to a standard CSV with header: `userId,movieId,rating,timestamp`.

## Download Flow

1. Check if `{data_dir}/{name}/ratings.csv` exists → if yes, skip to step 6.
2. Download zip from registry URL using `urllib.request.urlretrieve`.
3. Extract zip contents to `{data_dir}/{name}/raw/` using `zipfile`.
4. Read the raw ratings file with `pandas.read_csv()` using the registry's `sep`, `columns`, and `header` config.
5. Write the standardized CSV to `{data_dir}/{name}/ratings.csv`.
6. Delete the temporary zip file.
7. Construct and return `Dataset("{data_dir}/{name}/ratings.csv", **kwargs)`.

## Progress Display

- If `tqdm` is installed: show a download progress bar (consistent with existing training progress bar pattern).
- If `tqdm` is not installed: print `Downloading {name}...` and `Download complete.` messages.

Uses the same optional tqdm pattern already established in the project.

## Code Changes

### New file: `literec/data/downloader.py`

Contains:
- `DATASETS` registry dict
- `load_dataset(name, data_dir, **kwargs)` function
- `available_datasets()` function
- Internal helpers for download, extraction, and conversion

### Modified file: `literec/__init__.py`

Add exports:
```python
from .data.downloader import load_dataset, available_datasets
```

### Modified file: `pyproject.toml`

No changes needed. All functionality uses stdlib (`urllib.request`, `zipfile`, `pathlib`) plus existing `pandas` dependency.

## Testing

### New file: `tests/test_downloader.py`

| Test | Description |
|------|-------------|
| `test_available_datasets` | Returns correct sorted list of names |
| `test_invalid_name_raises` | `ValueError` with helpful message for unknown names |
| `test_convert_tab_separated` | TSV (ml-100k format) converts to correct CSV |
| `test_convert_double_colon` | `::` separated (ml-1m format) converts to correct CSV |
| `test_convert_csv_with_header` | CSV with header (ml-25m format) converts correctly |
| `test_cache_skip` | No download when `ratings.csv` already exists |
| `test_load_dataset_returns_dataset` | Returns `Dataset` with valid `n_users`, `n_items` |
| `test_kwargs_passthrough` | `min_rating`, `split` etc. forwarded to `Dataset()` |

Tests mock the download step to avoid network dependency. Format conversion tests use small synthetic data files.

## Design Decisions

1. **Independent function over class method:** `load_dataset()` is a standalone function, not `Dataset.load()`. Keeps `Dataset` focused on data processing; downloading is a separate concern.
2. **Local `./data/` over home directory:** Files are visible in the project directory, easy to inspect and version-control (via `.gitignore`).
3. **Preserve raw files:** Users can inspect original data. The `raw/` subdirectory keeps things organized.
4. **No new dependencies:** Uses stdlib for download/extraction, existing pandas for format conversion.
5. **Cache by file existence:** Simple check for `ratings.csv` — no version tracking or checksums. Sufficient for stable, versioned datasets like MovieLens.
6. **Hardcoded registry:** A dict is sufficient for 4 MovieLens variants. Can be refactored to a class-based approach when more dataset families are added.
