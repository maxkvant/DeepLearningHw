import torch.nn as nn
import torch
import numpy as np
from tensorboardX import SummaryWriter

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, model, device, train_dataloader, test_dataloader=None, summaryWriter=SummaryWriter(), criterion=nn.CrossEntropyLoss()):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.device = device
        self.model = model
        self.summaryWriter = summaryWriter

    def train(self, epochs, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []

        for epoch in range(1, epochs + 1):
            self.summaryWriter.add_text(tag='progress', text_string="epoch {}/{}".format(epoch, epochs))
            for batch_id, (image, label) in enumerate(self.train_dataloader):
                optimizer.zero_grad()

                label, image = label.to(self.device), image.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, label)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            self.summaryWriter.add_scalar('train_loss', np.average(losses), epoch)
            if self.test_dataloader is not None:
                self.summaryWriter.add_scalar('test_loss', self._average_loss(self.test_dataloader), epoch)

    def _average_loss(self, dataloader):
        with torch.no_grad():
            losses = []
            for batch_id, (image, label) in enumerate(dataloader):
                if (batch_id > 200):
                    break
                label, image = label.to(self.device), image.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, label)
                losses.append(loss.item())
        return np.average(losses)