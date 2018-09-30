import torch.nn as nn
import torch
import numpy as np
from tensorboardX import SummaryWriter


class Trainer:
    def init(self, model, device, train_dataloader, summaryWriter, test_dataloader=None, criterion=nn.CrossEntropyLoss):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.device = device
        self.model = model
        self.summaryWriter = summaryWriter

    def train(self, epochs, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_losses = []
        test_losses = [] if self.test_dataloader is not None else None
        losses = []

        for epoch in range(1, epochs + 1):
            self.summaryWriter.add_text(tag='progress', text="epoch {}/{}".format(epoch, epochs))
            for batch_id, (image, label) in enumerate(self.train_dataloader):
                optimizer.zero_grad()

                label, image = label.to(self.device), image.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, label)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            train_losses.append(np.average(losses))
            self.summaryWriter.add_scalar('train_loss', np.average(losses))
            if self.test_dataloader is not None:
                self.summaryWriter.add_scalar('test_loss', self._average_loss(self.test_dataloader))

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