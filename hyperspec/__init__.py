import os
from pathlib import Path

DATA_PATH = os.path.expanduser('~/.hyperspec/data')
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
