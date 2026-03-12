import pytest


@pytest.fixture
def tiny_csv(tmp_path):
    """Create a tiny CSV with 3 users, 5 items, enough interactions for LOO split."""
    data = (
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n"
        "1,20,4.0,200\n"
        "1,30,3.0,300\n"
        "1,40,5.0,400\n"
        "1,50,4.0,500\n"
        "2,10,4.0,100\n"
        "2,20,5.0,200\n"
        "2,30,3.0,300\n"
        "2,40,4.0,400\n"
        "2,50,5.0,500\n"
        "3,10,5.0,100\n"
        "3,20,4.0,200\n"
        "3,30,5.0,300\n"
        "3,40,3.0,400\n"
        "3,50,4.0,500\n"
    )
    path = tmp_path / "ratings.csv"
    path.write_text(data)
    return str(path)
