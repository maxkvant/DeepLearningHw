import resnext
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


def test_network():
    classes = 3
    model = resnext.resnext50_32(in_channels=1, num_classes=classes)
    device = torch.device('cpu')
    batch_size = 2

    train_dataset = TensorDataset(
        torch.rand(batch_size * 3, 1, 256, 256),
        torch.tensor(torch.randint(0, classes, (batch_size * 3,)), dtype=torch.long)
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    trainer = resnext.Trainer(model, device, train_dataloader)
    trainer.train(10)