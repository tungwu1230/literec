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


def test_loo_split_few_interactions(tmp_path):
    """Users with < 3 interactions go to train only, no crash."""
    data = (
        "userId,movieId,rating,timestamp\n"
        # user 1: 5 interactions (normal)
        "1,10,5.0,100\n1,20,4.0,200\n1,30,3.0,300\n1,40,5.0,400\n1,50,4.0,500\n"
        # user 2: 2 interactions (too few for LOO)
        "2,10,4.0,100\n2,20,5.0,200\n"
        # user 3: 5 interactions (normal)
        "3,10,5.0,100\n3,20,4.0,200\n3,30,5.0,300\n3,40,3.0,400\n3,50,4.0,500\n"
    )
    path = tmp_path / "ratings.csv"
    path.write_text(data)
    ds = Dataset(str(path), min_interactions=2, split="loo")
    # user with 2 interactions should be in train only
    user2_id = ds._user_map[2]
    assert user2_id not in ds.test_data
    assert user2_id not in ds.valid_data
    assert len(ds.train_data[user2_id]) == 2


def test_train_matrix_no_leakage(tiny_csv):
    ds = Dataset(tiny_csv, split="loo")
    for uid in range(ds.n_users):
        train_items_from_matrix = set(ds.train_matrix[uid].indices)
        train_items_from_data = set(ds.train_data[uid])
        assert train_items_from_matrix == train_items_from_data
