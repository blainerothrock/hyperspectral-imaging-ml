import torch, gin
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from hyperspec.datasets import indian_pine_split
from hyperspec.model import HybridSN
import numpy as np
from ignite.engine import create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite_utils import create_engine, prepare_batch
from torchsummary import summary

@gin.configurable()
def train(learning_rate, weight_decay, max_epochs):
    train_dl, test_dl = indian_pine_split()

    device = torch.device('cuda')

    model = HybridSN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = create_engine(model, optimizer, loss_fn, device)
    eval = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(), 'nll': Loss(loss_fn)},
        prepare_batch=prepare_batch,
        device=device
    )

    summary(model, next(iter(train_dl))[0].unsqueeze(1).shape[1:])

    checkpoint_handler = ModelCheckpoint('./models', 'hybridsn', n_saved=1, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})

    writer = SummaryWriter()

    @trainer.on(Events.ITERATION_COMPLETED)
    def report_iter_loss(trainer):
        writer.add_scalar('iter/train_loss', trainer.state.output, trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def report_training_epoch(trainer):
        print('\nEpoch {}'.format(trainer.state.epoch))
        eval.run(train_dl)
        metrics = eval.state.metrics
        print('  - training:')
        print('    - loss: {:.2f}'.format(metrics['nll']))
        print('    - acc:  {:.2f}'.format(metrics['accuracy']))

        writer.add_scalar('loss/train', metrics['nll'], trainer.state.epoch)
        writer.add_scalar('accuracy/train', metrics['accuracy'], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def report_testing_epoch(trainer):
        eval.run(test_dl)
        metrics = eval.state.metrics
        print('  - testing:')
        print('    - loss: {:.2f}'.format(metrics['nll']))
        print('    - acc:  {:.2f}'.format(metrics['accuracy']))

        writer.add_scalar('loss/test', metrics['nll'], trainer.state.epoch)
        writer.add_scalar('accuracy/test', metrics['accuracy'], trainer.state.epoch)

    trainer.run(train_dl, max_epochs=max_epochs)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    train()
