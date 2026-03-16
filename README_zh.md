# Lite-Rec

輕量推薦系統框架，專為學術研究設計。以最少的依賴取代 RecBole，專注於協同過濾模型，支援 Python 3.11+ 與 PyTorch。

## 特色

- **3 種模型**：BPR（矩陣分解）、LightGCN、NGCF
- **完整排序評估**：Recall、NDCG、MRR、Hit、Precision @K
- **自動裝置偵測**：CUDA / MPS / CPU
- **Leave-one-out 與隨機切分**
- **Early stopping** 與最佳模型 checkpoint
- **可重現**：透過 seed 控制
- **僅 4 個依賴**：torch、numpy、scipy、pandas

## 安裝

```bash
# 從 GitHub 安裝（推薦用於 Colab）
pip install git+https://github.com/tungwu1230/literec.git

# 從本地安裝
git clone https://github.com/tungwu1230/literec.git
cd literec
pip install -e .

# 開發模式
pip install -e ".[dev]"
```

## 快速開始

```python
from literec import Dataset, LightGCN, Trainer

dataset = Dataset("ratings.csv", min_rating=3.5, split="loo")
model = LightGCN(dataset, n_layers=3, emb_size=64)
trainer = Trainer(model, dataset, epochs=100, lr=1e-3, topk=[10, 20])
result = trainer.fit()
# result = {'Recall@10': 0.15, 'NDCG@10': 0.08, ...}
```

## 模型

| 模型 | 說明 | 主要超參數 |
|------|------|-----------|
| `BPR` | 矩陣分解 + BPR 損失函數 | `emb_size`、`reg_weight` |
| `LightGCN` | 輕量圖卷積網路 | `emb_size`、`n_layers`、`reg_weight` |
| `NGCF` | 帶非線性變換的圖協同過濾 | `emb_size`、`n_layers`、`dropout`、`reg_weight` |

```python
from literec import BPR, LightGCN, NGCF

model = BPR(dataset, emb_size=64, reg_weight=1e-5)
model = LightGCN(dataset, emb_size=64, n_layers=3, reg_weight=1e-5)
model = NGCF(dataset, emb_size=64, n_layers=3, dropout=0.1, reg_weight=1e-5)
```

## 資料集

讀取 CSV 檔案，支援自訂欄位名稱：

```python
from literec import Dataset

# 預設欄位：userId, movieId, rating, timestamp
dataset = Dataset("ratings.csv", min_rating=3.5, split="loo")

# 自訂欄位名稱
dataset = Dataset(
    "data.csv",
    user_col="user_id",
    item_col="item_id",
    rating_col="score",
    time_col="ts",
    min_rating=0.0,
    min_interactions=5,  # 過濾互動次數 < 5 的使用者
    split="loo",         # "loo"（leave-one-out）或 "random"
)

print(dataset.n_users, dataset.n_items)
```

## 訓練器

```python
from literec import Trainer

trainer = Trainer(
    model, dataset,
    epochs=100,
    batch_size=2048,
    lr=1e-3,
    weight_decay=0.0,
    early_stop_patience=10,
    device="auto",       # "auto"、"cpu"、"cuda"
    seed=42,
    topk=[10, 20, 50],
    metrics=["recall", "ndcg", "hit"],
    eval_batch_size=256,
)
result = trainer.fit()
```

支援的指標：`recall`、`ndcg`、`mrr`、`hit`、`precision`

## CLI

```bash
python run.py --data ratings.csv --model lightgcn --epochs 100 --lr 0.001
python run.py --data ratings.csv --model bpr --min_rating 3.5 --topk 10 20 50
python run.py --data ratings.csv --model ngcf --n_layers 3 --dropout 0.1
```

## 專案結構

```
literec/
├── config.py              # LiteRecConfig dataclass
├── data/
│   ├── dataset.py         # CSV 載入、ID 重新映射、資料切分
│   └── dataloader.py      # Pairwise 負採樣
├── model/
│   ├── base.py            # AbstractRecommender
│   ├── bpr.py             # BPR
│   ├── lightgcn.py        # LightGCN
│   └── ngcf.py            # NGCF
├── training/
│   └── trainer.py         # 訓練迴圈、early stopping
└── evaluation/
    └── evaluator.py       # 完整排序評估指標
```

## 環境需求

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.26
- SciPy >= 1.12
- pandas >= 2.1
