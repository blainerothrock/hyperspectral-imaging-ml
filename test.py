import torch
from torch.utils.data import DataLoader
from hyperspec.datasets import IndianPineDataset
from hyperspec.model import HybridSN
from torchsummary import summary
import numpy as np

ds = IndianPineDataset()
dl = DataLoader(ds, batch_size=256, shuffle=True)

device = torch.device('cuda')


model = HybridSN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-06)
loss_fn = torch.nn.CrossEntropyLoss()

# summary(model, (1,25,25,30))

model.to(device)
model.train()

for epoch in range(100):
    print('epoch {}'.format(epoch+1))
    running_loss = []
    for batch in dl:
        data, label = batch
        n = data.numpy()
        # print('mean: {:.4f}, min: {:.4f}, max: {:.4f}, std: {:.4f}'.format(np.mean(n), np.min(n), np.max(n), np.std(n)))
        data = data.unsqueeze(1).float().to(device)
        label = label.to(torch.int64).to(device)

        optimizer.zero_grad()

        output = model(data)
        _loss = loss_fn(output, label)
        _loss.backward()
        optimizer.step()
        running_loss.append(_loss.cpu().detach().item())

    print('  - {}: Loss: {}'.format(epoch, np.mean(running_loss)))
    running_loss = []