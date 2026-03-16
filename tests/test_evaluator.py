import numpy as np
import torch
from literec.evaluation.evaluator import Evaluator


def test_recall_perfect_prediction():
    scores = torch.tensor([
        [10.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 10.0, 1.0, 1.0, 1.0],
    ])
    ground_truth = {0: [0], 1: [1]}
    train_mask = {0: [2], 1: [3]}

    evaluator = Evaluator(topk=[1, 5], metrics=["recall", "ndcg"])
    results = evaluator.compute(scores, ground_truth, train_mask)
    assert results["Recall@1"] == [1.0, 1.0]
    assert results["NDCG@1"] == [1.0, 1.0]


def test_recall_worst_prediction():
    scores = torch.tensor([
        [10.0, 1.0, 1.0, 1.0, 1.0],
    ])
    ground_truth = {0: [0]}
    train_mask = {0: [0]}

    evaluator = Evaluator(topk=[1], metrics=["recall"])
    results = evaluator.compute(scores, ground_truth, train_mask)
    assert results["Recall@1"] == [0.0]


def test_uneven_batches_give_correct_mean():
    """Verify that per-user scores avoid batch-averaging bias."""
    evaluator = Evaluator(topk=[1], metrics=["hit"])

    # Batch 1: 2 users, both hit
    scores1 = torch.tensor([[10.0, 1.0], [1.0, 10.0]])
    gt1 = {0: [0], 1: [1]}
    mask1 = {0: [], 1: []}
    r1 = evaluator.compute(scores1, gt1, mask1)

    # Batch 2: 1 user, miss
    scores2 = torch.tensor([[1.0, 10.0]])
    gt2 = {0: [0]}
    mask2 = {0: []}
    r2 = evaluator.compute(scores2, gt2, mask2)

    # Combine with extend (correct) vs append (wrong)
    all_scores = r1["Hit@1"] + r2["Hit@1"]  # [1.0, 1.0, 0.0]
    assert np.isclose(np.mean(all_scores), 2.0 / 3.0)
