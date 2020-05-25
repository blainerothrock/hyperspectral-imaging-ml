import pytest
from hyperspec.datasets import indian_pine_split
from hyperspec.datasets import IndianPineDataset
from torch.utils.data import DataLoader
import torch


class TestIndianPineDataset:

    # TODO: placeholder tests
    def test_initialization(self):
        K = 12
        window_size = 13
        batch_size = 1

        ds = IndianPineDataset(K=K, window_size=window_size)

        dl = DataLoader(ds, batch_size=batch_size, num_workers=1)
        X, y = next(iter(dl))

        assert isinstance(X, torch.Tensor), 'X should be a Tensor'
        assert X.shape == (batch_size, window_size, window_size, K)
        assert isinstance(y, torch.Tensor), 'y should be a Tensor'
        assert len(y) == batch_size

    def test_split(self):
        force_download = False
        K = 30
        window_size = 25
        train_split=0.7

        train_ds, test_ds = indian_pine_split(force_download=force_download, K=K, window_size=window_size, train_split=train_split)

        assert train_ds.img.data.shape[0] == train_ds.labels.data.shape[0]
        assert train_ds.img.data.shape[1] == train_ds.img.data.shape[2] == window_size
        assert train_ds.img.data.shape[3] == K

        assert test_ds.img.data.shape[0] == test_ds.labels.data.shape[0]
        assert test_ds.img.data.shape[1] == test_ds.img.data.shape[2] == window_size
        assert test_ds.img.data.shape[3] == K

        assert train_ds.img.data.shape[0] > test_ds.img.data.shape[0]
        assert int((train_ds.img.data.shape[0] + test_ds.img.data.shape[0]) * train_split) == train_ds.img.data.shape[0]
        assert int((train_ds.img.data.shape[0] + test_ds.img.data.shape[0]) * (1-train_split)) + 1 == test_ds.img.data.shape[0]








