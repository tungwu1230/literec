from __future__ import annotations

import numpy as np
import torch


class Evaluator:
    METRIC_NAMES = {
        "recall": "Recall", "ndcg": "NDCG", "mrr": "MRR",
        "hit": "Hit", "precision": "Precision",
    }

    def __init__(
        self,
        topk: list[int] | None = None,
        metrics: list[str] | None = None,
    ):
        self.topk = topk or [10, 20]
        self.metrics = [m.lower() for m in (metrics or ["recall", "ndcg"])]
        self.max_k = max(self.topk)

    def compute(
        self,
        scores: torch.Tensor,
        ground_truth: dict[int, list[int]],
        train_mask: dict[int, list[int]],
    ) -> dict[str, float]:
        for uid, items in train_mask.items():
            scores[uid, items] = float("-inf")

        _, topk_indices = torch.topk(scores, self.max_k, dim=1)

        results: dict[str, float] = {}
        for k in self.topk:
            topk_k = topk_indices[:, :k]
            for metric in self.metrics:
                display = self.METRIC_NAMES.get(metric, metric)
                key = f"{display}@{k}"
                fn = getattr(self, f"_{metric}")
                values = []
                for uid in ground_truth:
                    gt = set(ground_truth[uid])
                    pred = topk_k[uid].tolist()
                    values.append(fn(pred, gt))
                results[key] = float(np.mean(values)) if values else 0.0
        return results

    @staticmethod
    def _recall(pred: list[int], gt: set[int]) -> float:
        return len(set(pred) & gt) / len(gt) if gt else 0.0

    @staticmethod
    def _ndcg(pred: list[int], gt: set[int]) -> float:
        if not gt:
            return 0.0
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(pred) if item in gt)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), len(pred))))
        return dcg / ideal if ideal > 0 else 0.0

    @staticmethod
    def _mrr(pred: list[int], gt: set[int]) -> float:
        for i, item in enumerate(pred):
            if item in gt:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _hit(pred: list[int], gt: set[int]) -> float:
        return 1.0 if set(pred) & gt else 0.0

    @staticmethod
    def _precision(pred: list[int], gt: set[int]) -> float:
        return len(set(pred) & gt) / len(pred) if pred else 0.0
