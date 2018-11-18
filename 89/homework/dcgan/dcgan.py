import torch.nn as nn


class DCGenerator(nn.Module):
    def __init__(self, img_size, latent_size=100, out_channels=3, channels_inside=128):
        super(DCGenerator, self).__init__()

        self.latent_size = latent_size
        self.init_size = img_size // 8
        self.layers = channels_inside
        self.l1 = nn.Sequential(nn.Linear(latent_size, channels_inside * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(channels_inside),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_inside, channels_inside, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(channels_inside, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(channels_inside, channels_inside // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_inside // 2, 0.8),
            nn.Upsample(scale_factor=2),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels_inside // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z.view(-1, self.latent_size))
        out = out.view(out.shape[0], self.layers, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, img_size, in_channels=3, channels_inside=64):
        super(DCDiscriminator, self).__init__()

        self.img_size = img_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, channels_inside, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_inside),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels_inside, channels_inside * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_inside * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels_inside * 2, channels_inside * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_inside * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.activation = nn.Sequential(
            nn.Conv2d(channels_inside * 4, 1, kernel_size=(img_size // 8), stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv_layers(img)
        return self.activation(out)