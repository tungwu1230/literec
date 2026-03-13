from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractRecommender(nn.Module, ABC):
    @abstractmethod
    def calculate_loss(
        self, user: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def compute_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (all_user_emb, all_item_emb)."""
        ...

    def predict(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return user_emb @ item_emb.T
