import pytest

from literec.data.downloader import available_datasets, load_dataset


def test_available_datasets():
    result = available_datasets()
    assert result == ["ml-100k", "ml-1m", "ml-10m", "ml-25m"]


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset 'not-a-dataset'"):
        load_dataset("not-a-dataset")


from literec.data.downloader import _convert_raw_to_csv


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
