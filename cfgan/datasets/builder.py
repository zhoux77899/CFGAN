from functools import partial
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..datasets.molecule import MoleculeDataset
from ..datasets.residual import ResidualMoleculeDataset


def build_molecule_dataset(
    dataset_type: str,
    img_dir: str,
    psf_dir: str,
    suffix: str = ".tif",
    filter_args: Optional[dict] = None,
    normalization: bool = True,
    train: bool = True,
    pct: float = 0.8
) -> MoleculeDataset:
    """Build a PyTorch Dataset for molecular imaging data.

    Creates and returns a MoleculeDataset instance for loading and processing 
    molecular images and their corresponding Point Spread Function (PSF) data.

    :param dataset_type: Dataset type identifier (maintained for interface consistency)
    :param img_dir: Directory path containing molecular image files
    :param psf_dir: Directory path containing Point Spread Function (PSF) files
    :param suffix: File extension of image files, default is ".tif"
    :param filter_args: Dictionary of filtering parameters for CrossFilter
    :param normalization: Whether to apply min-max normalization to the data, default is True
    :param train: Whether to return training set data,  default is True
    :param pct: Proportion of the dataset to use for training (range [0,1]), default is 0.8
    :return: Configured MoleculeDataset instance compatible with PyTorch DataLoader
    """

    return MoleculeDataset(
        img_dir=img_dir,
        psf_dir=psf_dir,
        suffix=suffix,
        filter_args=filter_args,
        normalization=normalization,
        train=train,
        pct=pct
    )


def build_residual_dataset(
        dataset_type: str,
        img_lists: list[dict],
        normalization: bool = True,
        train: bool = True,
        pct: float = 0.8
) -> ResidualMoleculeDataset:
    """Build a PyTorch Dataset for residual molecule imaging data.

    Creates and returns a ResidualMoleculeDataset instance for loading and processing
    residual molecule images with their corresponding emitter count information. 

    :param dataset_type: Dataset type identifier (maintained for interface consistency)
    :param img_lists: List of dictionaries where keys are the number of residual emitters
                      and values are corresponding image tensors or numpy arrays
    :param normalization: Whether to apply min-max normalization to the data, default is True
    :param train: Whether to return training set data, default is True
    :param pct: Proportion of the dataset to use for training (range [0,1]), default is 0.8
    :return: Configured ResidualMoleculeDataset instance compatible with PyTorch DataLoader
    """

    return ResidualMoleculeDataset(
        img_lists=img_lists,
        normalization=normalization,
        train=train,
        pct=pct,
    )


def build_dataloader(
    dataset: Dataset,
    batches_per_gpu: int,
    workers_per_gpu: int,
    num_gpus: int = 1,
    shuffle: bool = True,
    seed: int = None,
    pin_memory: bool = False,
    **kwargs
) -> DataLoader:
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    :param dataset: A PyTorch dataset.
    :param batches_per_gpu: Batch size of each GPU.
    :param workers_per_gpu: Subprocesses to use for data loading for each GPU.
    :param num_gpus: Number of GPUs. Only used in non-distributed training.
    :param shuffle: Whether to shuffle the data at every epoch. Default: True.
    :param seed: Seed to be used. Default: None.
    :param pin_memory: Whether to use pin_memory in DataLoader. Default: False.
    :param kwargs: Any keyword argument to be used to initialize DataLoader.
    :return: A PyTorch dataloader.
    """

    batch_size = num_gpus * batches_per_gpu
    num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        seed=seed
    ) if seed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        pin_memory=pin_memory,
        **kwargs
    )


def worker_init_fn(worker_id: int, num_workers: int, seed: int) -> None:
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker + worker_id + user_seed

    :param worker_id: Worker id.
    :param num_workers: Number of workers.
    :param seed: The random seed to use.
    """

    worker_seed = num_workers + worker_id + seed
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
