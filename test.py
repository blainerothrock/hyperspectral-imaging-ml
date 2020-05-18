import torch
from torch.utils.data import DataLoader
from hyperspec.datasets import IndianPineDataset
from hyperspec.model import HybridSN
from torchsummary import summary

ds = IndianPineDataset()
dl = DataLoader(ds, batch_size=256, shuffle=True)

device = torch.device('cuda')

batch = next(iter(dl))

model = HybridSN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn = torch.nn.CrossEntropyLoss()

summary(model, (1,30,25,25))

for epoch in range(100):
    print('epoch {}'.format(epoch+1))
    datas, labels = batch
    for i in range(256):
        data = datas[i].view(30, 25, 25).unsqueeze(0).unsqueeze(1).float().to(device)
        # data = datas[i].unsqueeze(0).unsqueeze(1).float().to(device)
        label = labels[i].unsqueeze(0).to(torch.int64).to(device)

        model.train()
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

    print('  - {}: Loss: {}'.format(epoch, loss.cpu().item()))