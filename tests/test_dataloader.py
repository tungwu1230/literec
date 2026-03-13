import torch
from literec.data.dataset import Dataset
from literec.data.dataloader import TrainDataLoader


def test_dataloader_output_shape(tiny_csv):
    ds = Dataset(tiny_csv)
    loader = TrainDataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    users, pos_items, neg_items = batch
    assert users.shape == pos_items.shape == neg_items.shape
    assert len(users) <= 4


def test_negative_sampling(tiny_csv):
    ds = Dataset(tiny_csv)
    loader = TrainDataLoader(ds, batch_size=64)
    for users, pos_items, neg_items in loader:
        for i in range(len(users)):
            uid = users[i].item()
            neg = neg_items[i].item()
            # Negative item must NOT be in user's positive set
            assert neg not in ds.train_data[uid]
