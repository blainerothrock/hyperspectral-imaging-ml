import os
import shutil

import torch
from hyperspec.utility.download import download_file, DatasetURL
from hyperspec.transforms import Compose, PCA, ImageTransform, ToTensor
from hyperspec import DATA_PATH
from torch.utils.data import Dataset
from scipy import io
import numpy as np


class IndianPineDataset(Dataset):
    """
    Indian Pine Hyperspectral dataset (Purdue University)

    Args:
        TODO:
    """

    def __init__(self, force_download=False, K=30, window_size=25):

        self.K = K
        self.window_size = window_size
        self.prepared_data_path = None
        self._length = 0
        self.img = []
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

        self.rgb_bands = (43, 21, 11)  # AVIRIS sensor

        self.gt = io.loadmat(self.gt_path)['indian_pines_gt']
        self.labels = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                       "Corn", "Grass-pasture", "Grass-trees",
                       "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                       "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                       "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                       "Stone-Steel-Towers"]

        self.preprocess()

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.labels[i]

    def preprocess(self):
        processed_path = os.path.join(DATA_PATH, 'ip/prepared/')
        self.prepared_data_path = os.path.join(processed_path, 'ip_prepared.pt')

        # TODO: provide a better check for if process data exists
        if os.path.isdir(processed_path) and len(os.listdir(processed_path)) > 0:
            data = torch.load(self.prepared_data_path)
            self.img = data['X']
            self.labels = data['y']
            return

        if os.path.isdir(processed_path):
            shutil.rmtree(processed_path)
        os.makedirs(processed_path)

        pca_tf = PCA(num_components=self.K, source='src', inplace=True)
        img_tf = ImageTransform(source='src', window_size=self.window_size, inplace=True)
        tensor_tf = ToTensor(source='src', inplace=True)

        data = {'src': (self.img, self.gt)}
        pca_tf(data)
        img_tf(data)
        tensor_tf(data)

        img, labels = data['src']
        self.img = img
        self.labels = labels

        to_save = {'X': img, 'y': labels}
        torch.save(to_save, open(self.prepared_data_path, 'wb'))
