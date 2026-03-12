# Lite-Rec Design Spec

> This spec supersedes `ARCHITECTURE.md`. Where they disagree, this document is authoritative.

## Overview

Lite-Rec is a lightweight recommendation framework replacing RecBole, focused on CF/MF/GNN models for academic research. Targets Python 3.12+, Colab-friendly, with only 4 dependencies.

## Dependencies

- `torch` — models & training
- `numpy` — numerical computation
- `scipy` — sparse matrices (adjacency matrix)
- `pandas` — CSV data loading

## Directory Structure

```
lite-rec/
├── literec/
│   ├── __init__.py              # Public API: Dataset, Trainer, BPR, LightGCN, NGCF
│   ├── config.py                # @dataclass config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # CSV loading, ID remapping, filtering, split
│   │   └── dataloader.py        # Training DataLoader + uniform negative sampling
│   ├── model/
│   │   ├── __init__.py
│   │   ├── base.py              # AbstractRecommender base class
│   │   ├── bpr.py               # BPR (Matrix Factorization)
│   │   ├── lightgcn.py          # LightGCN
│   │   └── ngcf.py              # NGCF
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop + early stopping + checkpoint + auto device
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py         # Full ranking: Recall, NDCG, MRR, Hit, Precision @K
├── run.py                        # CLI entry point
├── pyproject.toml                # Python 3.12+, 4 dependencies
└── tests/
    └── ...
```

## Config

```python
@dataclass
class LiteRecConfig:
    # Data
    data_path: str                          # CSV file path
    user_col: str = "userId"                # column name for user ID
    item_col: str = "movieId"              # column name for item ID
    rating_col: str = "rating"             # column name for rating
    time_col: str = "timestamp"            # column name for timestamp
    min_rating: float = 0.0                 # >= min_rating treated as positive
    min_interactions: int = 5               # users with fewer interactions are dropped
    split: Literal["loo", "random"] = "loo" # leave-one-out or random
    test_ratio: float = 0.1                 # for random split
    valid_ratio: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0              # optimizer weight_decay (NOT model L2 reg)
    early_stop_patience: int = 10
    device: str = "auto"                    # "auto" / "cpu" / "cuda"
    seed: int = 42                          # random seed for reproducibility

    # Evaluation
    topk: list[int] = field(default_factory=lambda: [10, 20])
    metrics: list[str] = field(default_factory=lambda: ["recall", "ndcg"])
    eval_batch_size: int = 256              # users per batch during evaluation
```

### Config usage

- `LiteRecConfig` holds training/evaluation parameters only. Model hyperparameters (`emb_size`, `n_layers`, etc.) are passed directly to model constructors.
- `Dataset` accepts data-related params directly: `Dataset("ratings.csv", min_rating=3.5, split="loo")`
- `Trainer` accepts training/eval params directly: `Trainer(model, dataset, epochs=100, lr=1e-3)`
- Alternatively, pass `config=LiteRecConfig(...)` to `Trainer` for centralized configuration. Direct kwargs override config values.
- `weight_decay` is for the optimizer only. Models that need L2 regularization (BPR) compute it internally via a `reg_weight` hyperparameter to avoid double penalization.

### Reproducibility

`seed` controls all random sources: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, and Python `random.seed`. Set once at `Trainer` initialization.

## Data Layer

### Dataset

- Reads CSV with configurable column names (default: `userId, movieId, rating, timestamp`)
- Remaps raw IDs to contiguous integers (`user_id: 0~N`, `item_id: 0~M`)
- Filters by `min_rating`, converting to implicit feedback (interaction = 1)
- Drops users with fewer than `min_interactions` interactions (default 5) after filtering
- Split strategies:
  - `loo` (default): sort by timestamp per user, last interaction → test, second-to-last → valid, rest → train. Requires >= 3 interactions per user (guaranteed by `min_interactions >= 3`).
  - `random`: split by ratio (default 8:1:1)
- Exposes properties:
  - `n_users`, `n_items` — counts after remapping
  - `train_matrix` — train-only interaction matrix (scipy CSR sparse), used by GNN models for adjacency matrix construction. Does NOT include valid/test to prevent data leakage.
  - `train_data`, `valid_data`, `test_data` — dict of `{user_id: [item_ids]}`

### DataLoader

- Produces `(user, pos_item, neg_item)` triplets per batch
- Uniform negative sampling: ensures neg_item is not in the user's positive set (looked up per sample)
- Standard PyTorch `DataLoader`, supports `num_workers` (positive sets are stored in the dataset object, pickled to workers)

## Model Layer

### AbstractRecommender (base class, extends `nn.Module`)

Three abstract methods:
- `calculate_loss(user, pos_item, neg_item) → loss` — for training
- `compute_all_embeddings() → (user_emb, item_emb)` — runs the full forward pass once, returns all user and item embeddings. For BPR this is a simple lookup; for GNN models this runs the graph propagation.
- `predict(user_emb, item_emb) → scores` — computes `user_emb @ item_emb.T`, default implementation in base class.

The evaluator calls `compute_all_embeddings()` once, then slices user embeddings per batch. This avoids running GNN propagation multiple times.

### BPR (Matrix Factorization)

- User/Item embedding lookup tables
- Score = dot product of user and item embeddings
- Loss = `-log_sigmoid(pos_score - neg_score)` + L2 regularization (via `reg_weight` param, default 1e-5)
- Model-specific hyperparams: `emb_size` (default 64), `reg_weight` (default 1e-5)

### LightGCN

- Constructs normalized adjacency matrix from `dataset.train_matrix` at model init (once, not per forward pass)
- Normalization: symmetric `D^{-1/2} A D^{-1/2}` where A is the user-item bipartite adjacency matrix
- Stored as torch sparse tensor on the model's device
- K layers of light graph convolution (neighbor aggregation only, no nonlinearity)
- Final embedding = mean of all layer embeddings (layer 0 through layer K)
- BPR loss with `reg_weight`
- Model-specific hyperparams: `emb_size` (default 64), `n_layers` (default 3), `reg_weight` (default 1e-5)

### NGCF

- Same adjacency matrix construction and normalization as LightGCN
- Each layer: linear transform + LeakyReLU + feature interaction term
- Dropout between layers
- BPR loss with `reg_weight`
- Model-specific hyperparams: `emb_size` (default 64), `n_layers` (default 3), `dropout` (default 0.1), `reg_weight` (default 1e-5)

## Training

### Trainer

- Sets random seeds at initialization for reproducibility
- Auto device detection: `cuda` → `mps` → `cpu`
- Standard loop: each epoch runs train → evaluate on valid set
- Early stopping: monitors primary metric on valid set (default `Recall@10`), stops after `patience` epochs without improvement
- Checkpoint: saves best model state_dict to a temp file, auto-loads after training completes, cleaned up on exit
- `fit()` returns test set evaluation results after training completes (loads best checkpoint, evaluates on test set)
- Prints loss and valid metrics per epoch to stdout

## Evaluation

### Evaluator

- Full ranking: calls `model.compute_all_embeddings()` once, then processes users in batches (`eval_batch_size`, default 256)
- Per batch: computes `user_emb_batch @ all_item_emb.T`, masks items in train/valid (set to `-inf`), computes top-K metrics
- Supported metrics: Recall@K, NDCG@K, MRR@K, Hit@K, Precision@K
- Scale expectation: works well for datasets up to ~100K items. For larger datasets, evaluation will be slower but functional with the batching strategy.

## CLI (`run.py`)

Minimal CLI for running experiments from the command line:

```bash
python run.py --data ratings.csv --model lightgcn --epochs 100 --lr 0.001
```

- `--data` (required): path to CSV file
- `--model` (required): one of `bpr`, `lightgcn`, `ngcf`
- All other config and model params can be passed as `--key value` flags
- Prints final test results to stdout

## Target API

```python
from literec import Dataset, BPR, LightGCN, NGCF, Trainer

# Load data
dataset = Dataset("ratings.csv", min_rating=3.5, split="loo")

# Create model
model = LightGCN(dataset, n_layers=3, emb_size=64)

# Train & evaluate
trainer = Trainer(model, dataset, epochs=100, lr=1e-3, topk=[10, 20])
result = trainer.fit()
print(result)  # {'Recall@10': 0.15, 'NDCG@10': 0.08, ...}
```

## Design Decisions

1. **CSV only, no .inter** — users work with raw data files directly, with configurable column names
2. **Implicit feedback with configurable min_rating** — flexible threshold for positive samples
3. **min_interactions filter** — ensures enough data per user for LOO split and meaningful evaluation
4. **Leave-one-out as default split** — aligns with BPR/LightGCN/NGCF paper conventions
5. **Full ranking evaluation** — no sampled metrics bias, matches current academic standards
6. **dataclass config** — IDE-friendly, type-safe, with sensible defaults
7. **Auto device detection** — seamless CPU/GPU/MPS switching for Colab
8. **Single `compute_all_embeddings()` call** — avoids redundant GNN propagation during evaluation
9. **Separate `reg_weight` from optimizer `weight_decay`** — prevents double L2 penalization
10. **`train_matrix` is train-only** — prevents data leakage into GNN adjacency matrix
11. **Seed-based reproducibility** — all random sources controlled by a single seed
