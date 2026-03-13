import torch
from literec.data.dataset import Dataset
from literec.model.bpr import BPR


def test_bpr_loss(tiny_csv):
    ds = Dataset(tiny_csv)
    model = BPR(ds, emb_size=16)
    users = torch.tensor([0, 1])
    pos = torch.tensor([0, 1])
    neg = torch.tensor([2, 3])
    loss = model.calculate_loss(users, pos, neg)
    assert loss.shape == ()
    assert loss.item() > 0


def test_bpr_embeddings(tiny_csv):
    ds = Dataset(tiny_csv)
    model = BPR(ds, emb_size=16)
    u_emb, i_emb = model.compute_all_embeddings()
    assert u_emb.shape == (ds.n_users, 16)
    assert i_emb.shape == (ds.n_items, 16)
