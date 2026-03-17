import pytest

from literec.data.downloader import available_datasets, load_dataset


def test_available_datasets():
    result = available_datasets()
    assert result == ["ml-100k", "ml-1m", "ml-10m", "ml-25m"]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset 'not-a-dataset'"):
        load_dataset("not-a-dataset")
