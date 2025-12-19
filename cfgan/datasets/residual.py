import random

import torch
from torch.utils.data import Dataset


class ResidualMoleculeDataset(Dataset):
    """
    Residual molecule image dataset.
    """

    def __init__(
            self,
            img_lists: list[dict],
            normalization: bool = False,
            train: bool = True,
            pct: float = 0.8
    ) -> None:
        super(ResidualMoleculeDataset, self).__init__()
        self.img_lists = img_lists
        self.normalization = normalization
        self.train = train
        self.pct = pct if 0 <= pct <= 1 else 0.8

        self.emitters, self.images = self.split_images()

    @staticmethod
    def rearrange_images(img_lists: list[dict]) -> list[dict]:
        return random.sample(img_lists, len(img_lists))

    @staticmethod
    def normalize(image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def split_images(self) -> tuple[list[int], list[torch.Tensor]]:
        emitter_set = []
        image_set = []

        for img_dict in self.rearrange_images(self.img_lists):
            for num_residual_emitters, img_tensor in img_dict.items():
                try:
                    assert isinstance(img_tensor, torch.Tensor)
                except AssertionError:
                    img_tensor = torch.from_numpy(img_tensor)
                emitter_set.append(num_residual_emitters)
                image_set.append(img_tensor)

        assert len(emitter_set) == len(image_set)

        if self.train:
            emitters = emitter_set[:int(self.pct * len(emitter_set))]
            images = image_set[:int(self.pct * len(image_set))]
        else:
            emitters = emitter_set[int(self.pct * len(emitter_set)):]
            images = image_set[int(self.pct * len(image_set)):]

        return emitters, images

    def __len__(self) -> int:
        """Total number of samples of data.
        """
        assert len(self.emitters) == len(self.images)
        return len(self.emitters)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Get training/test data after pipeline.

        :param index: Index of data.
        :return: `tuple[int, torch.Tensor]`: Training/test data.
        """

        emitter = self.emitters[index]
        image = self.images[index]

        if self.normalization:
            image = self.normalize(image)

        return emitter, image
