from __future__ import annotations

import torch
import torch.nn as nn

from literec.model.base import AbstractRecommender


class BPR(AbstractRecommender):
    def __init__(self, dataset, emb_size: int = 64, reg_weight: float = 1e-5):
        super().__init__()
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(self.n_users, emb_size)
        self.item_embedding = nn.Embedding(self.n_items, emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_all_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def calculate_loss(self, user, pos_item, neg_item):
        u_emb = self.user_embedding(user)
        p_emb = self.item_embedding(pos_item)
        n_emb = self.item_embedding(neg_item)

        pos_score = (u_emb * p_emb).sum(dim=1)
        neg_score = (u_emb * n_emb).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

        reg_loss = self.reg_weight * (
            u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)
        ) / u_emb.shape[0]

        return bpr_loss + reg_loss
