import os, requests
from tqdm import tqdm

class IndianPineDataset:
    """
    Indian Pine Hyperspectral dataset (Purdue University)

    Args:
        TODO:
    """

    def __init__(self):
        pass

    def download_data(self):
        download_path = os.path.expanduser('~/.hyperspec/data/')
        url = 'https://purr.purdue.edu/publications/1947/serve/1?render=archive'

        file_path = os.path.join(download_path, 'indian_pines.zip')
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
