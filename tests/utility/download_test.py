import pytest, os
from hyperspec.utility.download import download_file, DatasetURL, purge_data
from hyperspec import DATA_PATH


def test_download():
    # A very small file
    url = 'https://blainerothrock-public.s3.amazonaws.com/seisml/mars/event_images/S0105a_raw.png'
    name = 'test.png'

    file_path = download_file(url, name)

    assert os.path.isfile(file_path), 'downloaded file should exits'
    os.remove(file_path)


# def test_purge_data():
#     purge_data()
#     assert not os.path.isdir(DATA_PATH)
