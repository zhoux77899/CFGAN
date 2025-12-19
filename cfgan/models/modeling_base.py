import torch
import torch.nn as nn


class CNNForwardBlock(nn.Module):
    """Forward Block for CNN Discriminator
    """

    def __init__(self, in_channels: int, out_channels_a: int, out_channels_b: int):
        super(CNNForwardBlock, self).__init__()
        self.conv2_a = nn.Conv2d(in_channels, out_channels_a, (3, 3), padding='same')
        self.bn2_a = nn.BatchNorm2d(out_channels_a)
        self.conv2_b = nn.Conv2d(out_channels_a, out_channels_b, (3, 3), stride=(2, 2), padding=1)
        self.bn2_b = nn.BatchNorm2d(out_channels_b)
        self.act_fn = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2_a(input)
        x = self.bn2_a(x)
        x = self.act_fn(x)
        x = self.conv2_b(x)
        x = self.bn2_b(x)
        x = self.act_fn(x)

        return x
