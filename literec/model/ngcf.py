from __future__ import annotations

import torch
import torch.nn as nn

from literec.model.base import AbstractRecommender
from literec.utils import build_norm_adj


class NGCF(AbstractRecommender):
    def __init__(
        self,
        dataset,
        emb_size: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        reg_weight: float = 1e-5,
    ):
        super().__init__()
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(self.n_users, emb_size)
        self.item_embedding = nn.Embedding(self.n_items, emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        for _ in range(n_layers):
            self.W1.append(nn.Linear(emb_size, emb_size))
            self.W2.append(nn.Linear(emb_size, emb_size))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm_adj = build_norm_adj(dataset.train_matrix, self.n_users, self.n_items)

    def compute_all_embeddings(self):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]
        adj = self.norm_adj.to(all_emb.device)

        for i in range(self.n_layers):
            neighbor_emb = torch.sparse.mm(adj, all_emb)
            sum_emb = self.W1[i](all_emb + neighbor_emb)
            bi_emb = self.W2[i](all_emb * neighbor_emb)
            all_emb = self.leaky_relu(sum_emb + bi_emb)
            all_emb = self.dropout(all_emb)
            embs.append(all_emb)

        all_emb = torch.cat(embs, dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    def calculate_loss(self, user, pos_item, neg_item):
        u_emb_all, i_emb_all = self.compute_all_embeddings()
        u_emb = u_emb_all[user]
        p_emb = i_emb_all[pos_item]
        n_emb = i_emb_all[neg_item]

        pos_score = (u_emb * p_emb).sum(dim=1)
        neg_score = (u_emb * n_emb).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

        u_raw = self.user_embedding(user)
        p_raw = self.item_embedding(pos_item)
        n_raw = self.item_embedding(neg_item)
        reg_loss = self.reg_weight * (
            u_raw.norm(2).pow(2) + p_raw.norm(2).pow(2) + n_raw.norm(2).pow(2)
        ) / u_emb.shape[0]

        return bpr_loss + reg_loss
