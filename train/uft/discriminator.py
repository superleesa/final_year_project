import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, bn: bool = True):
        super(DiscriminatorBlock, self).__init__()
        filter_size = 4
        self.conv = nn.Conv2d(in_filters, out_filters, filter_size, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_filters) if bn else None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.lrelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Assume size is H=330 and W=440

        # 1) increase filter size by 2**i
        # 2) stride by 2 i.e. size reduces by factor of 2 at each step
        self.model = nn.Sequential(
            DiscriminatorBlock(3, 64, bn=False),  # Output size: (64, 165, 220)
            DiscriminatorBlock(64, 128),  # Output size: (128, 82, 110)
            DiscriminatorBlock(128, 256),  # Output size: (256, 41, 55)
            DiscriminatorBlock(256, 512),  # Output size: (512, 20, 27)
            nn.Conv2d(512, 1, 4, padding=0),  # Output size: (1, 9, 12)
            nn.Flatten(),
            nn.Linear(408, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)
