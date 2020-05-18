import torch
from torch.utils.data import DataLoader
from hyperspec.datasets import IndianPineDataset
from hyperspec.model import HybridSN
from torchsummary import summary

ds = IndianPineDataset()
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = HybridSN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for data, label in dl:
        data = data.unsqueeze(1).float()
        label = label.long()

        model.train()

        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        print('Loss: {}'.format(loss.item()))