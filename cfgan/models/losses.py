import torch
import torchvision.models as vmodels
import pytorch_msssim


def init_device(device: str = None) -> torch.device:
    if device:
        device = 'cuda:0' if torch.cuda.is_available() and device.startswith('cuda') else 'cpu'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


class GeneratorLoss(torch.nn.Module):
    """Loss function for training Generator.
    """

    def __init__(
            self,
            l1_weight: float,
            ssim_weight: float = 0,
            vgg_weight: float = 0,
            device: str = None
    ) -> None:
        super(GeneratorLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.vgg_weight = vgg_weight
        self.device = init_device(device)

    def l1_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        loss_func = torch.nn.L1Loss()
        return loss_func(input, target)

    def ssim_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        loss_func = pytorch_msssim.ssim
        return 1 - loss_func(input, target, data_range=1, size_average=True)

    def vgg_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        # load VGG model
        vgg_model = vmodels.vgg19(weights=vmodels.VGG19_Weights.DEFAULT).to(self.device)
        vgg_model.eval()
        vgg_criterion = torch.nn.MSELoss().to(self.device)

        # get only first channel
        input = input[:, 0:1, :, :]
        target = target[:, 0:1, :, :]

        # convert to RGB
        input_rgb = torch.cat([input, input, input], dim=1).to(self.device)
        target_rgb = torch.cat([target, target, target], dim=1).to(self.device)

        # calculate loss
        input_vgg = vgg_model(input_rgb)
        target_vgg = vgg_model(target_rgb)

        return vgg_criterion(input_vgg, target_vgg)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        batch_size = input.size(0)
        loss = 0.0 + torch.full(
            (batch_size,),
            0.0,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1_loss(input, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(input, target)
        if self.vgg_weight > 0:
            loss += self.vgg_weight * self.vgg_loss(input, target)

        return loss.mean()


class DiscriminatorLoss(torch.nn.Module):
    """Loss function for training Discriminator.
    """

    def __init__(self, device: str = None) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.device = init_device(device)

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor
    ) -> torch.Tensor:
        loss = (real - 1.0) ** 2 + fake ** 2
        loss = loss.squeeze().to(self.device)
        return loss.mean()


class TerminatorLoss(torch.nn.Module):
    """Loss function for training Terminator.
    """

    def __init__(self, device: str = None) -> None:
        super(TerminatorLoss, self).__init__()
        self.device = init_device(device)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        loss_func = torch.nn.MSELoss()
        return loss_func(input, target)
