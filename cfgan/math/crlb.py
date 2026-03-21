import numpy as np
import torch

from cfgan.common.utils import get_image_dimension


def normalize_to_probability_distribution(
    imgs: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Normalize PSF images to a probability distribution.

    This function normalizes the input 3D image tensor so that the sum of all its
    elements equals 1.

    Args:
        imgs: Input 3D array of shape (nz, ny, nx).

    Returns:
        The normalized array.
    """
    assert get_image_dimension(imgs) == 3, "`imgs` must be a $z$-stack of images"

    if isinstance(imgs, np.ndarray):
        return imgs / imgs.sum()
    if isinstance(imgs, torch.Tensor):
        return imgs / imgs.sum()

    raise ValueError(f"Unknown imgs type: {type(imgs)}.")


class CRLB:
    """
    Calculate the Cramer-Rao Lower Bound (CRLB) for 3D localization.

    The CRLB provides a lower bound on the variance of any unbiased estimator.
    For single molecule localization, it gives the theoretical limit on localization
    precision.

    Args:
        psf_imgs: 3D PSF array of shape (nz, ny, nx)
        delta: Tuple of (dx, dz) pixel sizes
        eps: Small value for numerical stability (default: 1e-6)
    """

    def __init__(
        self,
        psf_imgs: np.ndarray | torch.Tensor,
        delta: tuple[float, float],
        eps: float = 1e-6,
    ) -> None:
        assert get_image_dimension(psf_imgs) == 3, "`psf_imgs` must be a $z$-stack of images"
        assert len(delta) == 2, "`delta` must be a 2-tuple of (dx, dz)"

        self.psf_imgs = normalize_to_probability_distribution(psf_imgs)
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

    def grad(
        self,
        direction: str,
        psf_imgs: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray | torch.Tensor:
        """
        Calculate the gradient of the CRLB with respect to the given direction.
        """
        if psf_imgs is None:
            psf_imgs = self.psf_imgs

        if direction == "x":
            grad = (psf_imgs[:, :, 2:] - psf_imgs[:, :, :-2]) / (2 * self.dx)
            return self.pad(grad, (1, 1, 0, 0, 0, 0))
        if direction == "y":
            grad = (psf_imgs[:, 2:, :] - psf_imgs[:, :-2, :]) / (2 * self.dx)
            return self.pad(grad, (0, 0, 1, 1, 0, 0))
        if direction == "z":
            grad = (psf_imgs[2:, :, :] - psf_imgs[:-2, :, :]) / (2 * self.dz)
            return self.pad(grad, (0, 0, 0, 0, 1, 1))

        raise NotImplementedError(f"CRLB.grad() does not support direction {direction}")

    def forward(
        self,
        total_photons: int,
        background_photons: int,
        mode: str,
    ) -> np.ndarray | torch.Tensor:
        psf_imgs = self.psf_imgs * total_photons
        denominator = self.maximum(psf_imgs + background_photons)

        if mode == "lateral":
            fx = self.maximum((self.grad("x", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fy = self.maximum((self.grad("y", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fx + 1 / fy)
        if mode == "axial":
            fz = self.maximum((self.grad("z", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fz)
        if mode == "radial":
            fx = self.maximum((self.grad("x", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fy = self.maximum((self.grad("y", psf_imgs) ** 2 / denominator).sum((1, 2)))
            fz = self.maximum((self.grad("z", psf_imgs) ** 2 / denominator).sum((1, 2)))
            return self.sqrt(1 / fx + 1 / fy + 1 / fz)

        raise NotImplementedError(f"CRLB.forward() does not support mode {mode}")
