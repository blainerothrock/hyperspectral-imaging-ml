import pytest
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




