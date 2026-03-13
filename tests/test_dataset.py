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


def test_loo_split(tiny_csv):
    ds = Dataset(tiny_csv, split="loo")
    for uid in range(ds.n_users):
        assert len(ds.test_data[uid]) == 1
        assert len(ds.valid_data[uid]) == 1
        assert len(ds.train_data[uid]) == 3  # 5 total - 1 test - 1 valid


def test_random_split(tiny_csv):
    ds = Dataset(tiny_csv, split="random")
    for uid in range(ds.n_users):
        total = len(ds.train_data[uid]) + len(ds.valid_data[uid]) + len(ds.test_data[uid])
        assert total == 5


def test_train_matrix_no_leakage(tiny_csv):
    ds = Dataset(tiny_csv, split="loo")
    for uid in range(ds.n_users):
        train_items_from_matrix = set(ds.train_matrix[uid].indices)
        train_items_from_data = set(ds.train_data[uid])
        assert train_items_from_matrix == train_items_from_data
