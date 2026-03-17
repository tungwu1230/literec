import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from literec.data.dataset import Dataset
from literec.data.downloader import _convert_raw_to_csv, available_datasets, load_dataset


def test_available_datasets():
    result = available_datasets()
    assert result == ["ml-100k", "ml-1m", "ml-10m", "ml-25m"]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset 'not-a-dataset'"):
        load_dataset("not-a-dataset")


def test_convert_tab_separated(tmp_path):
    """TSV format (ml-100k style)."""
    raw = tmp_path / "u.data"
    raw.write_text("1\t10\t5.0\t100\n2\t20\t4.0\t200\n")
    out = tmp_path / "ratings.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep="\t",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=None,
    )

    lines = out.read_text().strip().split("\n")
    assert lines[0] == "userId,movieId,rating,timestamp"
    assert lines[1] == "1,10,5.0,100"
    assert lines[2] == "2,20,4.0,200"


def test_convert_double_colon(tmp_path):
    """:: delimited format (ml-1m style)."""
    raw = tmp_path / "ratings.dat"
    raw.write_text("1::10::5.0::100\n2::20::4.0::200\n")
    out = tmp_path / "ratings.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep="::",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=None,
    )

    lines = out.read_text().strip().split("\n")
    assert lines[0] == "userId,movieId,rating,timestamp"
    assert lines[1] == "1,10,5.0,100"


def test_convert_csv_with_header(tmp_path):
    """Already-CSV format (ml-25m style) uses shutil.copy."""
    raw = tmp_path / "ratings.csv"
    raw.write_text("userId,movieId,rating,timestamp\n1,10,5.0,100\n2,20,4.0,200\n")
    out = tmp_path / "output.csv"

    _convert_raw_to_csv(
        raw_path=raw,
        csv_path=out,
        sep=",",
        columns=["userId", "movieId", "rating", "timestamp"],
        header=0,
    )

    assert out.read_text() == raw.read_text()


def _create_fake_zip(zip_path, inner_file, content):
    """Helper to create a zip containing a single file."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(inner_file, content)


def _mock_urlretrieve(url, filename, reporthook=None):
    """Mock that creates a zip mimicking ml-1m format."""
    content = "1::10::5.0::100\n1::20::4.0::200\n1::30::3.0::300\n"
    content += "1::40::5.0::400\n1::50::4.0::500\n"
    content += "2::10::4.0::100\n2::20::5.0::200\n2::30::3.0::300\n"
    content += "2::40::4.0::400\n2::50::5.0::500\n"
    content += "3::10::5.0::100\n3::20::4.0::200\n3::30::5.0::300\n"
    content += "3::40::3.0::400\n3::50::4.0::500\n"
    _create_fake_zip(Path(filename), "ml-1m/ratings.dat", content)


def test_load_dataset_returns_dataset(tmp_path):
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        ds = load_dataset("ml-1m", data_dir=tmp_path)

    assert isinstance(ds, Dataset)
    assert ds.n_users == 3
    assert ds.n_items == 5


def test_kwargs_passthrough(tmp_path):
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        ds = load_dataset("ml-1m", data_dir=tmp_path, min_rating=4.5, min_interactions=3)

    # min_rating=4.5 filters most interactions, min_interactions=3 drops users
    assert ds.n_users == 0


def test_cache_skip(tmp_path):
    """Second call should not re-download."""
    call_count = 0

    def counting_urlretrieve(url, filename, reporthook=None):
        nonlocal call_count
        call_count += 1
        _mock_urlretrieve(url, filename, reporthook)

    with patch("literec.data.downloader.urllib.request.urlretrieve", counting_urlretrieve):
        load_dataset("ml-1m", data_dir=tmp_path)
        load_dataset("ml-1m", data_dir=tmp_path)

    assert call_count == 1


def test_bad_zip_cleans_up(tmp_path):
    """Corrupted zip: temp files cleaned up, no partial state."""
    def bad_urlretrieve(url, filename, reporthook=None):
        Path(filename).write_bytes(b"this is not a zip")

    with patch("literec.data.downloader.urllib.request.urlretrieve", bad_urlretrieve):
        with pytest.raises(zipfile.BadZipFile):
            load_dataset("ml-1m", data_dir=tmp_path)

    # No zip or temp csv left
    assert not (tmp_path / "ml-1m.zip").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv.tmp").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv").exists()


def test_conversion_failure_cleans_up(tmp_path):
    """Failed conversion: temp csv cleaned up."""
    with patch("literec.data.downloader.urllib.request.urlretrieve", _mock_urlretrieve):
        with patch(
            "literec.data.downloader._convert_raw_to_csv",
            side_effect=RuntimeError("conversion failed"),
        ):
            with pytest.raises(RuntimeError, match="conversion failed"):
                load_dataset("ml-1m", data_dir=tmp_path)

    assert not (tmp_path / "ml-1m.zip").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv.tmp").exists()
    assert not (tmp_path / "ml-1m" / "ratings.csv").exists()


def test_public_import():
    """load_dataset and available_datasets accessible from top-level package."""
    from literec import load_dataset as ld, available_datasets as ad
    assert callable(ld)
    assert callable(ad)
