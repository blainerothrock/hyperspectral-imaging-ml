import pytest
import torch
from torch.utils.data import DataLoader
from hyperspec.datasets import IndianPineDataset
from hyperspec.model import HybridSN
from torchsummary import summary
import numpy as np
from scipy import stats


class TestHybridSN:

    def test_input_dist(self):
        ds = IndianPineDataset()
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        data, _ = next(iter(dl))
        data = data.numpy()
        mean = np.mean(data)
        std = np.mean(data)

        tol = 0.1
        assert mean < tol, 'mean should be close to zero'
        assert (std - 1.0) < tol, 'standard deviation should be close to one'

    def test_loss(self):
        batch_size = 32
        epochs = 5

        ds = IndianPineDataset()
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = HybridSN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-06)
        loss_fn = torch.nn.NLLLoss()

        batch = next(iter(dl))

        running_loss = []
        y = np.array(list(range(epochs * batch_size)))
        for epoch in range(epochs):
            datas, labels = batch
            for i in range(batch_size):
                data = datas[i].unsqueeze(0).unsqueeze(1).float()
                label = labels[i].unsqueeze(0).to(torch.int64)

                optimizer.zero_grad()

                output = model(data)
                _loss = loss_fn(output, label)
                _loss.backward()
                optimizer.step()
                running_loss.append(_loss.cpu().detach().item())

        running_loss = np.array(running_loss)
        slope, _, _, _, _ = stats.linregress(running_loss, y)

        assert slope < 0, 'loss should be decreasing'