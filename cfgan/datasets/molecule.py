import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

from ..common.utils import get_image_size
from ..denoise.filter import CrossFilter


FILTER_ARGS = {
    'fraction': (0.15, 0.60),
    'rotation': -22.5,
    'types': 'ri',
    'gaussian_radius': 1.25
}


class MoleculeDataset(Dataset):
    """Molecule image dataset.
    """

    def __init__(
        self,
        img_dir: str,
        psf_dir: str,
        suffix: str = '.tif',
        filter_args: Optional[dict] = None,
        normalization: bool = True,
        train: bool = True,
        pct: float = 0.8
    ) -> None:
        super(MoleculeDataset, self).__init__()
        self.img_dir = img_dir
        self.psf_dir = psf_dir
        self.suffix = suffix
        self.filter_args = (
            FILTER_ARGS
            if not filter_args
            else {
                'fraction': filter_args.pop('fraction', FILTER_ARGS['fraction']),
                'rotation': filter_args.pop('rotation', FILTER_ARGS['rotation']),
                'types': filter_args.pop('types', FILTER_ARGS['types']),
                'gaussian_radius': filter_args.pop('gaussian_radius', FILTER_ARGS['gaussian_radius']),
            }
        )
        self.normalization = normalization
        self.train = train
        self.pct = pct if 0 <= pct <= 1 else 0.8
        self.max_size = 256

        self.num_types = len(os.listdir(os.path.abspath(self.img_dir)))
        self.imgs, self.psfs = self.load_images_from_root()

    def rearrange_file(self, file_list: List[str]) -> List[str]:
        """Rearrange the filename in list according to a specified interval

        Args:
            file_list (list[str]): List of filenames.

        Returns:
            list[str]: Rearranged list of filenames.
        """

        rearranged_file_list = []
        for i in range(self.num_types):
            rearranged_file_list += [file_list[j] for j in range(i, len(file_list), self.num_types)]
        return rearranged_file_list

    def set_images(self, file_list: List[str], filter: bool = False) -> torch.Tensor:
        image_array = []
        for file in self.rearrange_file(file_list):
            image = tiff.imread(file)
            cur_size = image.shape[-1]
            tar_size = min(cur_size, self.max_size)
            image = image[
                :,
                np.newaxis,
                (cur_size - tar_size) // 2:(cur_size + tar_size) // 2,
                (cur_size - tar_size) // 2:(cur_size + tar_size) // 2
            ]
            image_array.append(image)
        image_stack = np.vstack(image_array)

        if filter:
            cross_filter = CrossFilter(shape=get_image_size(image_stack), **self.filter_args)
            for i in range(len(image_stack)):
                image_stack[i, 0] = cross_filter.denoise(image_stack[i, 0])

        return torch.from_numpy(image_stack.astype(np.float32))

    def load_images_from_root(self) -> Tuple[torch.Tensor, torch.Tensor]:
        img_list = []
        for img_filename in os.listdir(os.path.abspath(self.img_dir)):
            if img_filename.endswith(self.suffix):
                img_list.append(os.path.abspath(os.path.join(os.path.abspath(self.img_dir), img_filename)))

        psf_list = []
        for psf_filename in os.listdir(os.path.abspath(self.psf_dir)):
            if psf_filename.endswith(self.suffix):
                psf_list.append(os.path.abspath(os.path.join(os.path.abspath(self.psf_dir), psf_filename)))

        img_image_set = self.set_images(img_list, filter=True)
        psf_image_set = self.set_images(psf_list, filter=False)

        if self.train:
            imgs = img_image_set[:int(self.pct * len(img_image_set))]
            psfs = psf_image_set[:int(self.pct * len(psf_image_set))]
        else:
            imgs = img_image_set[int(self.pct * len(img_image_set)):]
            psfs = psf_image_set[int(self.pct * len(psf_image_set)):]

        return imgs, psfs

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def __len__(self) -> int:
        """Total number of samples of data.
        
        Returns:
            int: Number of samples.
        """
        return len(self.imgs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training/test data after pipeline.

        Args:
            index (int): Index of data.

        Returns:
            tuple (img, psf): Training/test data.
        """

        img = self.imgs[index]
        psf = self.psfs[index]

        if self.normalization:
            img = self.normalize(img)
            psf = self.normalize(psf)

        return img, psf
