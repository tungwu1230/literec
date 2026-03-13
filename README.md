# Lite-Rec

Lightweight recommendation framework for academic research. A minimal replacement for RecBole, focused on collaborative filtering models with modern Python (3.11+) and PyTorch.

## Features

- **3 models**: BPR (Matrix Factorization), LightGCN, NGCF
- **Full ranking evaluation**: Recall, NDCG, MRR, Hit, Precision @K
- **Auto device**: CUDA / MPS / CPU auto detection
- **Leave-one-out & random split**
- **Early stopping** with best model checkpoint
- **Reproducible** via seed control
- **4 dependencies**: torch, numpy, scipy, pandas

## Installation

```bash
# From GitHub (recommended for Colab)
pip install git+https://github.com/tungwu1230/literec.git

# From local clone
git clone https://github.com/tungwu1230/literec.git
cd literec
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from literec import Dataset, LightGCN, Trainer

dataset = Dataset("ratings.csv", min_rating=3.5, split="loo")
model = LightGCN(dataset, n_layers=3, emb_size=64)
trainer = Trainer(model, dataset, epochs=100, lr=1e-3, topk=[10, 20])
result = trainer.fit()
# result = {'Recall@10': 0.15, 'NDCG@10': 0.08, ...}
```

## Models

| Model | Description | Key Hyperparameters |
|-------|-------------|-------------------|
| `BPR` | Matrix Factorization with BPR loss | `emb_size`, `reg_weight` |
| `LightGCN` | Light Graph Convolution Network | `emb_size`, `n_layers`, `reg_weight` |
| `NGCF` | Neural Graph Collaborative Filtering | `emb_size`, `n_layers`, `dropout`, `reg_weight` |

```python
from literec import BPR, LightGCN, NGCF

model = BPR(dataset, emb_size=64, reg_weight=1e-5)
model = LightGCN(dataset, emb_size=64, n_layers=3, reg_weight=1e-5)
model = NGCF(dataset, emb_size=64, n_layers=3, dropout=0.1, reg_weight=1e-5)
```

## Dataset

Reads CSV files with configurable column names:

```python
from literec import Dataset

# Default columns: userId, movieId, rating, timestamp
dataset = Dataset("ratings.csv", min_rating=3.5, split="loo")

# Custom column names
dataset = Dataset(
    "data.csv",
    user_col="user_id",
    item_col="item_id",
    rating_col="score",
    time_col="ts",
    min_rating=0.0,
    min_interactions=5,  # drop users with < 5 interactions
    split="loo",         # "loo" (leave-one-out) or "random"
)

print(dataset.n_users, dataset.n_items)
```

## Trainer

```python
from literec import Trainer

trainer = Trainer(
    model, dataset,
    epochs=100,
    batch_size=2048,
    lr=1e-3,
    weight_decay=0.0,
    early_stop_patience=10,
    device="auto",       # "auto", "cpu", "cuda"
    seed=42,
    topk=[10, 20, 50],
    metrics=["recall", "ndcg", "hit"],
    eval_batch_size=256,
)
result = trainer.fit()
```

Supported metrics: `recall`, `ndcg`, `mrr`, `hit`, `precision`

## CLI

```bash
python run.py --data ratings.csv --model lightgcn --epochs 100 --lr 0.001
python run.py --data ratings.csv --model bpr --min_rating 3.5 --topk 10 20 50
python run.py --data ratings.csv --model ngcf --n_layers 3 --dropout 0.1
```

## Project Structure

```
literec/
├── config.py              # LiteRecConfig dataclass
├── data/
│   ├── dataset.py         # CSV loading, ID remapping, split
│   └── dataloader.py      # Pairwise negative sampling
├── model/
│   ├── base.py            # AbstractRecommender
│   ├── bpr.py             # BPR
│   ├── lightgcn.py        # LightGCN
│   └── ngcf.py            # NGCF
├── training/
│   └── trainer.py         # Training loop, early stopping
└── evaluation/
    └── evaluator.py       # Full ranking metrics
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.26
- SciPy >= 1.12
- pandas >= 2.1
