import torch
from literec.data.dataset import Dataset
from literec.model.bpr import BPR
from literec.model.lightgcn import LightGCN
from literec.model.ngcf import NGCF

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


def test_lightgcn_loss(tiny_csv):
    ds = Dataset(tiny_csv)
    model = LightGCN(ds, emb_size=16, n_layers=2)
    users = torch.tensor([0, 1])
    pos = torch.tensor([0, 1])
    neg = torch.tensor([2, 3])
    loss = model.calculate_loss(users, pos, neg)
    assert loss.shape == ()
    assert loss.item() > 0


def test_lightgcn_embeddings(tiny_csv):
    ds = Dataset(tiny_csv)
    model = LightGCN(ds, emb_size=16, n_layers=2)
    u_emb, i_emb = model.compute_all_embeddings()
    assert u_emb.shape == (ds.n_users, 16)
    assert i_emb.shape == (ds.n_items, 16)


def test_lightgcn_propagation_changes_embeddings(tiny_csv):
    ds = Dataset(tiny_csv)
    model = LightGCN(ds, emb_size=16, n_layers=2)
    raw_user = model.user_embedding.weight.detach().clone()
    u_emb, _ = model.compute_all_embeddings()
    assert not torch.allclose(u_emb, raw_user)


def test_ngcf_loss(tiny_csv):
    ds = Dataset(tiny_csv)
    model = NGCF(ds, emb_size=16, n_layers=2, dropout=0.1)
    users = torch.tensor([0, 1])
    pos = torch.tensor([0, 1])
    neg = torch.tensor([2, 3])
    loss = model.calculate_loss(users, pos, neg)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ngcf_embeddings(tiny_csv):
    ds = Dataset(tiny_csv)
    model = NGCF(ds, emb_size=16, n_layers=2)
    u_emb, i_emb = model.compute_all_embeddings()
    assert u_emb.shape == (ds.n_users, 16 * 3)  # concat of 3 layers (0, 1, 2)
    assert i_emb.shape == (ds.n_items, 16 * 3)


def test_ngcf_dropout_effect(tiny_csv):
    ds = Dataset(tiny_csv)
    model = NGCF(ds, emb_size=16, n_layers=2, dropout=0.5)
    model.train()
    u1, _ = model.compute_all_embeddings()
    u2, _ = model.compute_all_embeddings()
    assert not torch.allclose(u1, u2)
