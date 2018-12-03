import argparse
import os
import logging
import torch

import torchvision.datasets as datasets
from torchvision import transforms
from torch.optim import Adam
from homework.vae import VAETrainer, loss_function, VAE


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    config = parser.parse_args()

    return config

def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cuda')

    model = VAE()
    model = model.to(device)

    def get_dataloader(train):
        transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(root=config.data_root, download=True,
                                   transform=transform, train=train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=4, pin_memory=True)
        return dataloader

    trainer = VAETrainer(model=model,
                         train_loader=get_dataloader(train=True),
                         test_loader=get_dataloader(train=False),
                         optimizer=Adam(model.parameters(), lr=1e-3),
                         loss_function=loss_function,
                         device=device)

    for epoch in range(config.epochs):
        trainer.train(epoch, config.log_metrics_every)
        trainer.test(epoch, config.batch_size, config.log_metrics_every)

if __name__ == '__main__':
    main()
