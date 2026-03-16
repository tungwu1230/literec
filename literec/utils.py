from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def build_norm_adj(
    interaction_matrix: sp.spmatrix, n_users: int, n_items: int
) -> torch.Tensor:
    """Build symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    n = n_users + n_items
    R = interaction_matrix.tocoo()
    rows = np.concatenate([R.row, R.col + n_users])
    cols = np.concatenate([R.col + n_users, R.row])
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    degree = np.array(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    indices = torch.tensor(np.array([norm.row, norm.col]), dtype=torch.long)
    values = torch.tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n))
