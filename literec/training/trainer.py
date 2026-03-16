from __future__ import annotations

import random

import numpy as np
import torch

from literec.data.dataloader import TrainDataLoader
from literec.evaluation.evaluator import Evaluator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        epochs: int = 100,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stop_patience: int = 10,
        device: str = "auto",
        seed: int = 42,
        topk: list[int] | None = None,
        metrics: list[str] | None = None,
        eval_batch_size: int = 256,
    ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.eval_batch_size = eval_batch_size

        _set_seed(seed)
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.evaluator = Evaluator(topk=topk, metrics=metrics)
        self.train_loader = TrainDataLoader(dataset, batch_size=batch_size)

    def fit(self) -> dict[str, float]:
        best_metric = 0.0
        patience_counter = 0
        best_state = None
        primary_key = f"Recall@{self.evaluator.topk[0]}"

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            loader = self.train_loader
            if tqdm is not None:
                loader = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
            for users, pos_items, neg_items in loader:
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(users, pos_items, neg_items)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            valid_results = self._run_evaluation(
                self.dataset.valid_data,
                mask_data=self.dataset.train_data,
            )

            print(
                f"Epoch {epoch + 1:>{len(str(self.epochs))}}/{self.epochs} | "
                f"Loss: {avg_loss:.4f} | {primary_key}: {valid_results.get(primary_key, 0.0):.4f}"
            )

            # Early stopping
            current = valid_results.get(primary_key, 0.0)
            if current > best_metric:
                best_metric = current
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best and run test
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        test_mask: dict[int, list[int]] = {}
        for uid in range(self.dataset.n_users):
            test_mask[uid] = (
                self.dataset.train_data.get(uid, [])
                + self.dataset.valid_data.get(uid, [])
            )

        test_results = self._run_evaluation(self.dataset.test_data, mask_data=test_mask)
        self._print_table("Test Results", test_results)
        return test_results

    def _print_table(self, title: str, results: dict[str, float]):
        """Print results as a formatted table grouped by metric."""
        # Parse keys like "Recall@20" into (metric, k)
        parsed: dict[str, dict[int, float]] = {}
        for key, val in results.items():
            metric, k_str = key.split("@")
            k = int(k_str)
            parsed.setdefault(metric, {})[k] = val

        metrics = list(parsed.keys())
        ks = sorted(next(iter(parsed.values())).keys())

        # Header
        print(f"\n{title}:")
        header = "".join(f"{m:>10}" for m in metrics)
        print(f"{'':>8}{header}")

        # Rows
        for k in ks:
            row = "".join(f"{parsed[m][k]:>10.4f}" for m in metrics)
            print(f"{'@'+ str(k):>8}{row}")

    @torch.no_grad()
    def _run_evaluation(
        self,
        ground_truth: dict[int, list[int]],
        mask_data: dict[int, list[int]],
    ) -> dict[str, float]:
        self.model.eval()
        u_emb_all, i_emb_all = self.model.compute_all_embeddings()

        all_results: dict[str, list[float]] = {}
        user_ids = list(ground_truth.keys())

        for start in range(0, len(user_ids), self.eval_batch_size):
            batch_uids = user_ids[start : start + self.eval_batch_size]
            uid_tensor = torch.tensor(batch_uids, device=self.device)
            u_emb = u_emb_all[uid_tensor]
            scores = self.model.predict(u_emb, i_emb_all)

            batch_gt = {i: ground_truth[uid] for i, uid in enumerate(batch_uids)}
            batch_mask = {i: mask_data.get(uid, []) for i, uid in enumerate(batch_uids)}

            batch_results = self.evaluator.compute(scores, batch_gt, batch_mask)
            for k, v_list in batch_results.items():
                all_results.setdefault(k, []).extend(v_list)

        return {k: float(np.mean(vs)) for k, vs in all_results.items()}
