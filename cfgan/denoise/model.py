import os
from typing import Union

import numpy as np
import torch
from safetensors.torch import load_file

from ..common.utils import get_image_dimension, set_image_dimension
from ..models.modeling_cfgan import UNetGenerator


IMAGE_DIMENSION = 4


class CFGANModel:
    def __init__(self, model_path: Union[str, os.PathLike]) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create model instance
        self.model = UNetGenerator()
        # Load weights from safetensors file using safetensors library
        weights = load_file(model_path)
        self.model.load_state_dict(weights)
        self.model = self.model.float().to(self.device)
        self.model.eval()

    @torch.no_grad()
    def denoise(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denoise a cross filtered image using CFGAN model.

        Args:
            img (``ndarray`` or ``Tensor``): ArrayLike image of shape [B, C, H, W]

        Returns:
            ``ndarray`` or ``Tensor``
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        dimension = get_image_dimension(img)
        img = set_image_dimension(img, IMAGE_DIMENSION).float().to(self.device)
        denoised_img = self.model(img).cpu()
        denoised_img = set_image_dimension(denoised_img, dimension)

        if isinstance(img, np.ndarray):
            denoised_img = denoised_img.numpy()

        return denoised_img
