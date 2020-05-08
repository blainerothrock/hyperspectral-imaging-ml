import requests, os, shutil
from tqdm import tqdm
from enum import Enum
from hyperspec import DATA_PATH


class DatasetURL(str, Enum):
    INDIAN_PINES_RAW_ZIP = 'https://purr.purdue.edu/publications/1947/serve/1?render=archive'
    INDIAN_PINES = 'http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat'
    INDIAN_PINES_CORRECTED = 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    INDIAN_PINES_GROUNDTRUTH = 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'


def download_file(url, name):
    download_path = os.path.expanduser(DATA_PATH)
    url = url

    file_path = os.path.join(download_path, name)
    with open(file_path, 'wb') as f:
        print('downloading %s' % file_path)
        res = requests.get(url, stream=True)
        total_size = int(res.headers.get('Content-Length'))

        pbar = tqdm(
            total=total_size,
            initial=0,
            unit='B',
            unit_scale=True,
            desc=file_path.split('/')[-1]
        )

        if total_size is None:
            f.write(res.content)
        else:
            for data in res.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(4096)
    pbar.close()

    return file_path


def purge_data():
    shutil.rmtree(DATA_PATH)
