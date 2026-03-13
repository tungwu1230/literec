from literec.data.dataset import Dataset


def test_load_csv(tiny_csv):
    ds = Dataset(tiny_csv)
    assert ds.n_users == 3
    assert ds.n_items == 5


def test_min_rating_filter(tiny_csv):
    ds = Dataset(tiny_csv, min_rating=4.0, min_interactions=3)
    assert ds.n_users == 3


def test_min_interactions_filter(tiny_csv):
    ds = Dataset(tiny_csv, min_rating=4.5, min_interactions=3)
    assert ds.n_users == 0


def test_id_remapping(tiny_csv):
    ds = Dataset(tiny_csv)
    assert ds.n_users == 3
    assert ds.n_items == 5
    for uid in ds.train_data:
        assert 0 <= uid < ds.n_users
    for uid in ds.train_data:
        for iid in ds.train_data[uid]:
            assert 0 <= iid < ds.n_items
