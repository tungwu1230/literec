from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class Dataset:
    def __init__(
        self,
        data_path: str,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
        time_col: str = "timestamp",
        min_rating: float = 0.0,
        min_interactions: int = 5,
        split: str = "loo",
        test_ratio: float = 0.1,
        valid_ratio: float = 0.1,
    ):
        self.split_method = split
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio

        # Load and filter
        df = pd.read_csv(data_path)
        df = df.rename(columns={
            user_col: "user", item_col: "item",
            rating_col: "rating", time_col: "time",
        })
        df = df[df["rating"] >= min_rating]

        # Drop users with too few interactions
        user_counts = df["user"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user"].isin(valid_users)]

        # Remap IDs to contiguous integers
        self._user_map = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
        self._item_map = {t: i for i, t in enumerate(sorted(df["item"].unique()))}
        df["user"] = df["user"].map(self._user_map)
        df["item"] = df["item"].map(self._item_map)

        self.n_users = len(self._user_map)
        self.n_items = len(self._item_map)

        if self.n_users == 0:
            self.train_data: dict[int, list[int]] = {}
            self.valid_data: dict[int, list[int]] = {}
            self.test_data: dict[int, list[int]] = {}
            self.train_matrix = csr_matrix((0, 0))
            return

        # Split
        if split == "loo":
            self.train_data, self.valid_data, self.test_data = self._split_loo(df)
        else:
            self.train_data, self.valid_data, self.test_data = self._split_random(df)

        # Build train-only sparse matrix
        rows, cols = [], []
        for uid, items in self.train_data.items():
            for iid in items:
                rows.append(uid)
                cols.append(iid)
        self.train_matrix = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )

    def _split_loo(self, df: pd.DataFrame):
        train, valid, test = {}, {}, {}
        for uid, group in df.sort_values("time").groupby("user"):
            items = group["item"].tolist()
            if len(items) < 3:
                train[uid] = items
                continue
            test[uid] = [items[-1]]
            valid[uid] = [items[-2]]
            train[uid] = items[:-2]
        return train, valid, test

    def _split_random(self, df: pd.DataFrame):
        train, valid, test = {}, {}, {}
        rng = np.random.default_rng(42)
        for uid, group in df.groupby("user"):
            items = group["item"].tolist()
            rng.shuffle(items)
            n = len(items)
            n_test = max(1, int(n * self.test_ratio))
            n_valid = max(1, int(n * self.valid_ratio))
            test[uid] = items[:n_test]
            valid[uid] = items[n_test:n_test + n_valid]
            train[uid] = items[n_test + n_valid:]
        return train, valid, test
