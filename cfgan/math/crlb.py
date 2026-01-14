from dataclasses import dataclass

import numpy as np
import torch

from cfgan.common.utils import get_image_dimension, normalize


def normalize_to_probability_distribution(
    imgs: np.ndarray | torch.Tensor,
    eps: float,
) -> np.ndarray | torch.Tensor:
    assert get_image_dimension(imgs) == 3, "`imgs` must be a $z$-stack of images"

    num_imgs = imgs.shape[0]
    per_image_normalized_imgs = normalize(imgs, eps=eps, per_image=True)

    if isinstance(per_image_normalized_imgs, np.ndarray):
        sum_per_img = per_image_normalized_imgs.reshape(num_imgs, -1).sum(axis=1, keepdims=True).reshape(num_imgs, 1, 1)
        return per_image_normalized_imgs / sum_per_img

    if isinstance(per_image_normalized_imgs, torch.Tensor):
        sum_per_img = per_image_normalized_imgs.view(num_imgs, -1).sum(dim=1, keepdim=True).view(num_imgs, 1, 1)
        return per_image_normalized_imgs / sum_per_img

    raise TypeError(f"CRLB.normalize_to_probability_distribution() does not support {type(imgs)}")


@dataclass
class FisherInformation:
    x: np.ndarray | torch.Tensor | None = None
    y: np.ndarray | torch.Tensor | None = None
    z: np.ndarray | torch.Tensor | None = None


class CRLB:
    """
    Calculate the Cramer-Rao Lower Bound (CRLB) for a given parameter.
    """

    def __init__(
        self,
        psf_imgs: np.ndarray | torch.Tensor,
        delta: tuple[float, float],
        eps: float = 1e-6,
    ) -> None:
        assert get_image_dimension(psf_imgs) == 3, "`psf_imgs` must be a $z$-stack of images"
        assert len(delta) == 2, "`delta` must be a 2-tuple of (dx, dz)"

        self.psf_imgs = normalize_to_probability_distribution(psf_imgs, eps=eps)
        self.dx, self.dz = delta
        self.eps = eps

    def __call__(
        self,
        total_photons: int,
        background_photons: int,
        mode: str,
    ) -> np.ndarray | torch.Tensor:
        return self.forward(total_photons, background_photons, mode)

    def pad(self, grad: np.ndarray | torch.Tensor, padding: tuple[int, ...]) -> np.ndarray | torch.Tensor:
        assert len(padding) == 6, "`padding` must be a 6-tuple"

        if isinstance(grad, np.ndarray):
            pad_width = tuple(zip(padding[::2], padding[1::2]))
            pad_width = pad_width[::-1]
            return np.pad(grad, pad_width, mode="edge")
        if isinstance(grad, torch.Tensor):
            # use `ReplicationPad3d` to simulate edge mode
            padded_grad = grad.unsqueeze(0).unsqueeze(0)  # [D, H, W] -> [1, 1, D, H, W]
            torch_pad_func = torch.nn.ReplicationPad3d(padding)
            padded_grad = torch_pad_func(padded_grad)
            return padded_grad.squeeze(0).squeeze(0)  # [1, 1, D, H, W] -> [D, H, W]

        raise NotImplementedError(f"CRLB.pad() does not support {type(grad)}")

    def maximum(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.maximum(x, self.eps)
        if isinstance(x, torch.Tensor):
            return torch.maximum(x, torch.full_like(x, self.eps))
        raise NotImplementedError(f"CRLB.maximum() does not support {type(x)}")

    def sqrt(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            return np.sqrt(x)
        if isinstance(x, torch.Tensor):
            return torch.sqrt(x)
        raise NotImplementedError(f"CRLB.sqrt() does not support {type(x)}")

    def grad(self, direction: str) -> np.ndarray | torch.Tensor:
        """
        Calculate the gradient of the CRLB with respect to the given direction.
        """
        edge = 2

        if direction == "x":
            grad = (self.psf_imgs[:, :, edge:] - self.psf_imgs[:, :, :-edge]) / (2 * self.dx)
            return self.pad(grad, (1, 1, 0, 0, 0, 0))
        if direction == "y":
            grad = (self.psf_imgs[:, edge:, :] - self.psf_imgs[:, :-edge, :]) / (2 * self.dx)
            return self.pad(grad, (0, 0, 1, 1, 0, 0))
        if direction == "z":
            grad = (self.psf_imgs[edge:, :, :] - self.psf_imgs[:-edge, :, :]) / (2 * self.dz)
            return self.pad(grad, (0, 0, 0, 0, 1, 1))

        raise NotImplementedError(f"CRLB.grad() does not support direction {direction}")

    def forward(
        self,
        total_photons: int,
        background_photons: int,
        mode: str,
    ) -> np.ndarray | torch.Tensor:
        denominator = self.psf_imgs * total_photons + background_photons
        denominator = self.maximum(denominator)

        fisher_info = FisherInformation()
        if mode in ["lateral", "radial"]:
            fisher_info.x = self.maximum((self.grad("x") ** 2 / denominator).sum((1, 2)))
            fisher_info.y = self.maximum((self.grad("y") ** 2 / denominator).sum((1, 2)))
        if mode in ["axial", "radial"]:
            fisher_info.z = self.maximum((self.grad("z") ** 2 / denominator).sum((1, 2)))

        if mode == "lateral":
            assert fisher_info.x is not None and fisher_info.y is not None
            return self.sqrt(self.sqrt(1 / fisher_info.x) ** 2 + self.sqrt(1 / fisher_info.y) ** 2)
        if mode == "axial":
            assert fisher_info.z is not None
            return self.sqrt(1 / fisher_info.z)
        if mode == "radial":
            assert fisher_info.x is not None and fisher_info.y is not None and fisher_info.z is not None
            return self.sqrt(
                (
                    self.sqrt(1 / fisher_info.x) ** 2
                    + self.sqrt(1 / fisher_info.y) ** 2
                    + self.sqrt(1 / fisher_info.z) ** 2
                )
            )
        raise NotImplementedError(f"CRLB.forward() does not support mode {mode}")
