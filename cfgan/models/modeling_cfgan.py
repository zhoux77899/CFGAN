from typing import Tuple

import torch
import torch.nn as nn

from .modeling_base import CNNForwardBlock


class UNetEncoderBlock(nn.Module):
    """Encoder Block for U-Net Generator"""

    def __init__(self, in_channels: int, out_channels_a: int, out_channels_b: int):
        super(UNetEncoderBlock, self).__init__()
        self.conv2_a = nn.Conv2d(in_channels, out_channels_a, (3, 3), padding="same")
        self.bn2_a = nn.BatchNorm2d(out_channels_a)  # tensorflow: eps=0.001, momentum=0.99
        self.conv2_b = nn.Conv2d(out_channels_a, out_channels_b, (3, 3), padding="same")
        self.bn2_b = nn.BatchNorm2d(out_channels_b)  # tensorflow: eps=0.001, momentum=0.99
        self.pool = nn.MaxPool2d((2, 2))
        self.act_fn = nn.ReLU()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv2_a(input)
        x = self.bn2_a(x)
        x = self.act_fn(x)
        x = self.conv2_b(x)
        x = self.bn2_b(x)
        x = self.act_fn(x)
        x_pool = self.pool(x)

        return x_pool, x


class UNetDecoderBlock(nn.Module):
    """Decoder Block for U-Net Generator"""

    def __init__(self, in_channels: int, out_channels_a: int, out_channels_b: int, out_channels_c: int):
        super(UNetDecoderBlock, self).__init__()
        self.conv2_a = nn.Conv2d(in_channels, out_channels_a, (3, 3), padding="same")
        self.bn2_a = nn.BatchNorm2d(out_channels_a)  # tensorflow: eps=0.001, momentum=0.99
        self.conv2_b = nn.Conv2d(out_channels_a, out_channels_b, (3, 3), padding="same")
        self.bn2_b = nn.BatchNorm2d(out_channels_b)  # tensorflow: eps=0.001, momentum=0.99
        self.con2transpose_c = nn.ConvTranspose2d(
            out_channels_b, out_channels_c, (3, 3), stride=(2, 2), padding=1, output_padding=1
        )
        self.bn2_c = nn.BatchNorm2d(out_channels_c)  # tensorflow: eps=0.001, momentum=0.99
        self.act_fn = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2_a(input)
        x = self.bn2_a(x)
        x = self.act_fn(x)
        x = self.conv2_b(x)
        x = self.bn2_b(x)
        x = self.act_fn(x)
        x = self.con2transpose_c(x)
        x = self.bn2_c(x)
        x = self.act_fn(x)

        return x


class UNetGenerator(nn.Module):
    """U-Net Generator"""

    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder1 = UNetEncoderBlock(1, 32, 32)
        self.encoder2 = UNetEncoderBlock(32, 64, 64)
        self.encoder3 = UNetEncoderBlock(64, 128, 128)
        self.encoder4 = UNetEncoderBlock(128, 256, 256)
        self.encoder5 = UNetEncoderBlock(256, 512, 512)
        self.decoder4 = UNetDecoderBlock(512, 512, 512, 256)
        self.decoder3 = UNetDecoderBlock(512, 256, 256, 128)
        self.decoder2 = UNetDecoderBlock(256, 128, 128, 64)
        self.decoder1 = UNetDecoderBlock(128, 64, 64, 32)
        self.conv_block = UNetEncoderBlock(64, 32, 32)
        self.output_block = nn.Conv2d(32, 1, (1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x, e1x = self.encoder1(input)  # [  n,   1, 256, 256] => [  n,  32, 128, 128]
        x, e2x = self.encoder2(x)  # [  n,  32, 128, 128] => [  n,  64,  64,  64]
        x, e3x = self.encoder3(x)  # [  n,  64,  64,  64] => [  n, 128,  32,  32]
        x, e4x = self.encoder4(x)  # [  n, 128,  32,  32] => [  n, 256,  16,  16]
        _, x = self.encoder5(x)  # [  n, 256,  16,  16] => [  n, 512,  16,  16]
        x = self.decoder4(x)  # [  n, 512,  16,  16] => [  n, 256,  32,  32]
        x = torch.cat((x, e4x), dim=1)  # [  n, 256,  32,  32] => [  n, 512,  32,  32]
        x = self.decoder3(x)  # [  n, 512,  32,  32] => [  n, 128,  64,  64]
        x = torch.cat((x, e3x), dim=1)  # [  n, 128,  64,  64] => [  n, 256,  64,  64]
        x = self.decoder2(x)  # [  n,  64, 128, 128] => [  n,  64, 128, 128]
        x = torch.cat((x, e2x), dim=1)  # [  n, 128, 128, 128] => [  n, 128, 128, 128]
        x = self.decoder1(x)  # [  n, 128, 128, 128] => [  n,  32, 256, 256]
        x = torch.cat((x, e1x), dim=1)  # [  n,  32, 256, 256] => [  n,  64, 256, 256]
        _, x = self.conv_block(x)  # [  n,  64, 256, 256] => [  n,  32, 256, 256]
        x = self.output_block(x)  # [  n,  32, 256, 256] => [  n,   1, 256, 256]

        return x


class CNNDiscriminator(nn.Module):
    """CNN Discriminator"""

    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.input_block = nn.Conv2d(1, 32, (3, 3), padding="same")
        self.forward1 = CNNForwardBlock(32, 64, 64)
        self.forward2 = CNNForwardBlock(64, 128, 128)
        self.forward3 = CNNForwardBlock(128, 256, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(32, 32))
        self.act_fn = nn.LeakyReLU(negative_slope=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_block(input)  # [  n,   1, 256, 256] => [  n,  32, 256, 256]
        x = self.act_fn(x)  # [  n,  32, 256, 256] => [  n,  32, 256, 256]
        x = self.forward1(x)  # [  n,  32, 256, 256] => [  n,  64, 128, 128]
        x = self.forward2(x)  # [  n,  64, 128, 128] => [  n, 128,  64,  64]
        x = self.forward3(x)  # [  n, 128,  64,  64] => [  n, 256,  32,  32]
        x = self.avg_pool2d(x)  # [  n, 256,  32,  32] => [  n, 256,   1,   1]
        x = self.fc1(x.permute(0, 2, 3, 1))  # [  n, 256,   1,   1] => [  n, 256,   1,   1]
        x = self.act_fn(x)  # [  n, 256,   1,   1] => [  n, 256,   1,   1]
        x = self.fc2(x)  # [  n, 256,   1,   1] => [  n,   1,   1,   1]
        x = self.sigmoid(x)  # [  n,   1,   1,   1] => [  n,   1,   1,   1]

        return x
