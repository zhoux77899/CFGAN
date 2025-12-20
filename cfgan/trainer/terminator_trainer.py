import os
import pickle
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cfgan.common.logging import logger
from cfgan.common.monitor import monitor_disk_usage, monitor_gpu_memory
from cfgan.config import TerminatorDatasetConfig, TrainerConfig
from cfgan.datasets.builder import build_dataloader, build_residual_dataset
from cfgan.models.losses import TerminatorLoss
from cfgan.models.modeling_terminator import Terminator

# set random seed for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
MANUAL_SEED = 42
torch.manual_seed(MANUAL_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)  # needed for reproducible results


def init_weights(m: torch.nn.Module) -> None:
    """
    Custom weights initialization called on model.

    :param m: The model.
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class TerminatorTrainer:
    def __init__(
        self: "TerminatorTrainer",
        img_lists: list,
        num_epochs: int,
        num_batches: int,
        num_workers: int,
        num_gpus: int = 1,
        learning_rate: float = 0.0002,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-08,
        save_dir: str | os.PathLike = None,
        temp_dir: str | os.PathLike = None,
    ) -> None:
        self.img_lists = img_lists
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.save_dir = save_dir
        self.temp_dir = temp_dir

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

        # create the dataset
        self.dataset = build_residual_dataset(
            dataset_type="residual",
            img_lists=self.img_lists,
            train=True,
        )
        self.dataloader = build_dataloader(
            dataset=self.dataset,
            batches_per_gpu=self.num_batches // self.num_gpus,
            workers_per_gpu=self.num_workers // self.num_gpus,
            num_gpus=self.num_gpus,
            shuffle=True,
            seed=MANUAL_SEED,
            pin_memory=False,
        )

        # create the Terminator and apply `init_weight` function to randomly initialize all weights
        self.net = Terminator().to(self.device)
        self.net.apply(init_weights)

        # initialize the loss function
        self.criterion = TerminatorLoss(device=f"{self.device.type}:{self.device.index}")

        # initialize the optimizers
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
        )

        # initialize the learning rate schedulers
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 1.0, 0.1, num_epochs - 1)

    @staticmethod
    def _set_progressbar() -> Callable:
        try:
            get_ipython().__class__.__name__  # noqa
        except NameError:
            try:
                from rich.progress import track as progressbar
            except ImportError:
                from tqdm import tqdm as progressbar
        else:
            from tqdm.notebook import tqdm as progressbar
        return progressbar

    def train(self) -> list:
        """Train Terminator."""

        # tensorboard summary writer
        summary_writer = SummaryWriter(os.path.join("runs", "terminator"))
        logger.info("Running `tensorboard --logdir=runs` from command line.")

        summary_writer.add_graph(self.net, torch.randn(1, 1, 256, 256).to(self.device))

        # lists to keep track of progress
        losses = []

        iters = 0

        # start training loop
        logger.info(f"Starting training loop on {self.device}")

        progressbar = self._set_progressbar()

        # for each epoch
        for epoch in progressbar(range(self.num_epochs), "Training..."):
            # for each batch in the dataloader
            for i, data in enumerate(self.dataloader):  # type: int, list[torch.Tensor]
                num_list = data[0] > 0
                image_list = data[1]
                residual_nums: torch.Tensor = num_list.to(torch.float32).to(self.device)
                residual_images: torch.Tensor = image_list.unsqueeze(1).to(torch.float32).to(self.device)

                # update Terminator
                self.net.zero_grad()
                output = self.net(residual_images)
                err = self.criterion(output.flatten(), residual_nums.flatten())
                err.backward()
                t_nums = output[0].item()
                self.optimizer.step()

                # output training states
                if (i + 1) % 10 == 0:
                    # monitor dis usage
                    disk_usage = monitor_disk_usage()
                    gpu_free_memory = monitor_gpu_memory(self.device.index)

                    print(
                        f"[{epoch + 1:03d}/{self.num_epochs:03d}][{i + 1:03d}/{len(self.dataloader):03d}]\t"
                        f"loss: {err.item():.4f}, "
                        f"t_nums: {t_nums:.4f} | "
                        f"disk usage: {disk_usage}, "
                        f"GPU free memory: {gpu_free_memory:.4f}"
                    )
                    summary_writer.add_scalars("training/loss", {"loss": err.item()}, iters)
                    summary_writer.add_scalars("training/output", {"t_nums": t_nums}, iters)
                    summary_writer.add_scalars("training/disk_usage", disk_usage, iters)
                    summary_writer.add_scalars("training/gpu_free_memory", {"gpu_free_memory": gpu_free_memory}, iters)
                    summary_writer.add_scalars(
                        "training/learning_rate", {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]}, iters
                    )

                # save losses for plotting
                losses.append(err.item())

                iters += 1

            # update learning rate schedulers
            self.scheduler.step()

            # save temporary checkpoints
            if (epoch + 1) % 50 == 0 and self.temp_dir:
                torch.save(self.net, os.path.join(f"{self.temp_dir}", f"t_epoch{epoch + 1}.pt"))

        # finish training loop
        summary_writer.close()
        logger.info("Finish training loop.")

        # save checkpoints
        if self.save_dir:
            torch.save(self.net.state_dict(), os.path.join(f"{self.save_dir}", "t.pt"))
            logger.info("Trained checkpoint saved.")

        return losses


if __name__ == "__main__":
    config_file = "../config.ini"
    dataset_config = TerminatorDatasetConfig.from_ini_file(config_file)
    trainer_config = TrainerConfig.from_ini_file(config_file)

    with open(dataset_config.residual_dir, "rb") as file:
        img_lists = pickle.load(file)

    trainer = TerminatorTrainer(
        img_lists,
        trainer_config.num_epochs,
        trainer_config.num_batches,
        trainer_config.num_workers,
        trainer_config.num_gpus,
        trainer_config.learning_rate,
        trainer_config.adam_beta1,
        trainer_config.adam_beta2,
        trainer_config.adam_eps,
        trainer_config.save_dir,
        trainer_config.temp_dir,
    )

    losses = trainer.train()
    np.save(os.path.join("..", "checkpoints", "terminator_losses.npy"), losses)
