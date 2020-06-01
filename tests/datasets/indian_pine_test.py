import pytest
from hyperspec.datasets import indian_pine_split
from hyperspec.datasets import IndianPineDataset
from torch.utils.data import DataLoader
import torch


class TestIndianPineDataset:

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
        train_dl, test_dl = indian_pine_split(batch_size=1)

        train_indices = train_dl.batch_sampler.sampler.indices
        test_indices = test_dl.batch_sampler.sampler.indices

        for idx in train_indices:
            assert idx not in test_indices, 'no index should overlap'








