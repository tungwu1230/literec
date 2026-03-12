from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LiteRecConfig:
    # Data
    data_path: str = ""
    user_col: str = "userId"
    item_col: str = "movieId"
    rating_col: str = "rating"
    time_col: str = "timestamp"
    min_rating: float = 0.0
    min_interactions: int = 5
    split: Literal["loo", "random"] = "loo"
    test_ratio: float = 0.1
    valid_ratio: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 10
    device: str = "auto"
    seed: int = 42

    # Evaluation
    topk: list[int] = field(default_factory=lambda: [10, 20])
    metrics: list[str] = field(default_factory=lambda: ["recall", "ndcg"])
    eval_batch_size: int = 256
