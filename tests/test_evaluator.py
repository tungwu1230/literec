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
    assert results["Recall@1"] == 1.0
    assert results["NDCG@1"] == 1.0


def test_recall_worst_prediction():
    scores = torch.tensor([
        [10.0, 1.0, 1.0, 1.0, 1.0],
    ])
    ground_truth = {0: [0]}
    train_mask = {0: [0]}

    evaluator = Evaluator(topk=[1], metrics=["recall"])
    results = evaluator.compute(scores, ground_truth, train_mask)
    assert results["Recall@1"] == 0.0
