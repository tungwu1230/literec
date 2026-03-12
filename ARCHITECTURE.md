# Lite-Rec 輕量推薦框架架構提案

## 目標
取代 RecBole，只保留 CF/MF/GNN 相關模型，適合學術研究、Colab 友善。

## 背景與需求
- 使用者為學術研究者，主要研究推薦系統
- 常用模型：CF、MF、GNN（如 BPR、LightGCN、NGCF）
- 主要使用環境：Google Colab
- 只需要標準的訓練 & 評估流程，不需要超參調優或分散式訓練
- RecBole 年久失修，依賴過時（scipy, pandas, torch.load 等問題不斷），94 個模型中只用到少數幾個

## 依賴（僅 4 個）
- `torch` — 模型 & 訓練
- `numpy` — 數值計算
- `scipy` — 稀疏矩陣（鄰接矩陣）
- `pandas` — 資料載入

不需要：`ray`、`tensorboard`、`colorlog`、`texttable` 等

## 目錄結構
```
lite-rec/
├── literec/
│   ├── config.py          # 單檔設定（dict-based，不需 YAML）
│   ├── data/
│   │   ├── dataset.py     # 載入 .inter 檔 → user/item interaction
│   │   └── dataloader.py  # 批次 + 負採樣
│   ├── model/
│   │   ├── base.py        # AbstractRecommender（calculate_loss, predict, full_sort_predict）
│   │   ├── bpr.py         # BPR (MF baseline)
│   │   ├── lightgcn.py    # LightGCN (GNN)
│   │   ├── ngcf.py        # NGCF (GNN)
│   │   └── ...            # 需要時再加
│   ├── trainer.py         # 訓練迴圈 + early stopping + checkpoint
│   ├── evaluator.py       # Recall, NDCG, MRR, Hit, Precision @K
│   ├── sampler.py         # 負採樣（uniform）
│   └── utils.py           # seed, logger, sparse matrix helpers
├── run.py                 # CLI 入口
└── setup.py
```

## 核心設計原則

| 項目 | RecBole | 新框架 |
|------|---------|--------|
| 模型數 | 94 個 | 只放需要的（BPR, LightGCN, NGCF...） |
| 設定系統 | 3 層 YAML merge | 單純 dict，可直接傳參 |
| 依賴 | ~15 個 | 4 個核心 |
| 模型註冊 | 動態掃描全部模組 | 明確 import，不需 magic |
| 超參調優 | 內建 Ray Tune | 不內建，需要時外掛 Optuna |
| 分散式 | 內建 DDP | 不內建 |
| Colab 使用 | 需 `pip install` 多個依賴 | `pip install` 秒裝即用 |

## 目標 API（Colab 使用方式）
```python
from literec import Trainer, LightGCN, Dataset

dataset = Dataset('ml-100k')
model = LightGCN(dataset, n_layers=3, emb_size=64)
trainer = Trainer(model, dataset, epochs=100, lr=0.001)

result = trainer.fit()
print(result)  # {'Recall@10': 0.15, 'NDCG@10': 0.08, ...}
```

## 可參考的 RecBole 原始碼
以下是從 RecBole 移植時值得參考的核心邏輯：

- **模型基底類別**: `recbole/model/abstract_recommender.py` — `GeneralRecommender` 的介面設計
- **LightGCN**: `recbole/model/general_recommender/lightgcn.py` — 鄰接矩陣建構 & GCN 傳播
- **BPR**: `recbole/model/general_recommender/bpr.py` — MF + BPR loss
- **NGCF**: `recbole/model/general_recommender/ngcf.py` — 帶非線性的 GCN
- **資料載入**: `recbole/data/dataset/dataset.py` — .inter 檔解析邏輯
- **負採樣**: `recbole/sampler/sampler.py` — uniform negative sampling
- **評估指標**: `recbole/evaluator/metrics.py` — Recall, NDCG, MRR 的計算方式
- **訓練迴圈**: `recbole/trainer/trainer.py` — early stopping, checkpoint 邏輯
