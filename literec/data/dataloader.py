from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class PairwiseTrainDataset(TorchDataset):
    def __init__(self, train_data: dict[int, list[int]], n_items: int):
        self.n_items = n_items
        self.pairs: list[tuple[int, int]] = []
        self.user_pos_items: dict[int, set[int]] = {}
        for uid, items in train_data.items():
            self.user_pos_items[uid] = set(items)
            for iid in items:
                self.pairs.append((uid, iid))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, pos = self.pairs[idx]
        neg = random.randint(0, self.n_items - 1)
        while neg in self.user_pos_items[uid]:
            neg = random.randint(0, self.n_items - 1)
        return uid, pos, neg


def TrainDataLoader(
    dataset,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> DataLoader:
    train_ds = PairwiseTrainDataset(dataset.train_data, dataset.n_items)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
