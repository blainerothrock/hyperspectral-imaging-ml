import os
import shutil
import gin

import torch
from hyperspec.utility.download import download_file, DatasetURL
from hyperspec.transforms import Compose, PCA, ImageTransform, ToTensor
from hyperspec import DATA_PATH
from torch.utils.data import Dataset
from scipy import io
import numpy as np


@gin.configurable()
def indian_pine_split(force_download=False, K=30, window_size=25, train_split=0.7):
    train_ds = IndianPineDataset(force_download, K, window_size, 'train', train_split)
    test_ds = IndianPineDataset(force_download, K, window_size, 'test', train_split)
    return train_ds, test_ds

class IndianPineDataset(Dataset):
    """
    Indian Pine Hyperspectral dataset (Purdue University)

    Args:
        TODO:
    """

    def __init__(self, force_download=False, K=30, window_size=25, mode='train', train_split=0.7):

        self.K = K
        self.window_size = window_size
        self.prepared_data_path = None
        self._length = 0
        self.img = []
        self.mode = mode
        self.train_split = train_split
        self.labels = None

        file_name = 'indian_pines_corrected.mat'
        self.corrected_path = os.path.join(DATA_PATH, file_name)
        if not os.path.isfile(self.corrected_path) or force_download:
            download_file(DatasetURL.INDIAN_PINES_CORRECTED.value, file_name)

        file_name = 'indian_pines_groundtruth.mat'
        self.gt_path = os.path.join(DATA_PATH, file_name)
        if not os.path.isfile(self.gt_path) or force_download:
            download_file(DatasetURL.INDIAN_PINES_GROUNDTRUTH.value, file_name)

        img = io.loadmat(self.corrected_path)
        self.img = img['indian_pines_corrected']

        self.labels = io.loadmat(self.gt_path)['indian_pines_gt']

        self.preprocess()

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.labels[i]

    def preprocess(self):
        processed_path = os.path.join(DATA_PATH, 'ip/prepared/')
        self.prepared_data_path = os.path.join(processed_path, 'ip_prepared_{}.pt'.format(self.mode))

        # TODO: provide a better check for if process data exists
        if os.path.isdir(processed_path) and len(os.listdir(processed_path)) > 0 and os.path.exists(self.prepared_data_path):
            data = torch.load(self.prepared_data_path)
            img = data['X']
            labels = data['y']

            if img.shape[0] == self.window_size and img.shape[2] == self.K:
                self.img = img
                self.labels = labels
                return

        if os.path.isfile(self.prepared_data_path):
            os.remove(self.prepared_data_path)
        if not os.path.isdir(processed_path):
            os.makedirs(processed_path)

        pca_tf = PCA(num_components=self.K, source='src', inplace=True)
        img_tf = ImageTransform(source='src', window_size=self.window_size, inplace=True)
        tensor_tf = ToTensor(source='src', inplace=True)

        data = {'src': (self.img, self.labels)}
        pca_tf(data)
        img_tf(data)
        tensor_tf(data)

        img, labels = data['src']

        offset = int(len(img) * self.train_split)
        if self.mode == 'train':
            self.img = img[: offset]
            self.labels = labels[: offset]
        elif self.mode == 'test':
            self.img = img[offset:]
            self.labels = labels[offset:]


        to_save = {'X': self.img, 'y': self.labels}
        torch.save(to_save, open(self.prepared_data_path, 'wb'))
