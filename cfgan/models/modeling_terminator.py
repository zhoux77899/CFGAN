import torch
import torch.nn as nn

from .modeling_base import CNNForwardBlock


class Terminator(nn.Module):
    """Terminator
    """

    def __init__(self) -> None:
        super().__init__()
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
        x = self.input_block(input)             # [  n,   1, 256, 256] => [  n,  32, 256, 256]
        x = self.act_fn(x)                      # [  n,  32, 256, 256] => [  n,  32, 256, 256]
        x = self.forward1(x)                    # [  n,  32, 256, 256] => [  n,  64, 128, 128]
        x = self.forward2(x)                    # [  n,  64, 128, 128] => [  n, 128,  64,  64]
        x = self.forward3(x)                    # [  n, 128,  64,  64] => [  n, 256,  32,  32]
        x = self.avg_pool2d(x)                  # [  n, 256,  32,  32] => [  n, 256,   1,   1]
        x = self.fc1(x.permute(0, 2, 3, 1))     # [  n, 256,   1,   1] => [  n, 256,   1,   1]
        x = self.act_fn(x)                      # [  n, 256,   1,   1] => [  n, 256,   1,   1]
        x = self.fc2(x)                         # [  n, 256,   1,   1] => [  n,   1,   1,   1]
        x = self.sigmoid(x)                     # [  n,   1,   1,   1] => [  n,   1,   1,   1]

        return x
