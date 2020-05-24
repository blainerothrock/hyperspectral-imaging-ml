import torch
from ignite.engine import Engine


def prepare_batch(batch, device, non_blocking):
    data, label = batch
    data = data.unsqueeze(1).float().to(device)
    label = label.to(torch.int64).to(device)

    return data, label

def create_engine(model, optimizer, loss_fn, device):
    model.to(device)

    def _update(engine, batch):
        model.train()
        data, label = prepare_batch(batch, device, non_blocking=False)

        optimizer.zero_grad()

        output = model(data)
        _loss = loss_fn(output, label)
        _loss.backward()
        optimizer.step()

        return _loss.item()

    return Engine(_update)
