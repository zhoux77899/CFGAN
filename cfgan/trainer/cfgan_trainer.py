import os
from typing import List, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cfgan.config import CFGANDatasetConfig, TrainerConfig
from cfgan.common.logging import logger
from cfgan.common.monitor import monitor_disk_usage, monitor_gpu_memory
from cfgan.datasets.builder import build_molecule_dataset, build_dataloader
from cfgan.models.losses import GeneratorLoss, DiscriminatorLoss
from cfgan.models.modeling_cfgan import UNetGenerator, CNNDiscriminator


# set random seed for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
manual_seed = 42
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True, warn_only=True)  # needed for reproducible results


def init_weights(m: torch.nn.Module) -> None:
    """Custom weights initialization called on model.

    Args:
        m (torch.nn.Module): The model.
    """

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class CFGANTrainer:
    def __init__(
            self,
            img_dir: str | os.PathLike,
            psf_dir: str | os.PathLike,
            num_epochs: int,
            num_batches: int,
            num_workers: int,
            learning_rate: float = 0.0002,
            adam_beta1: float = 0.9,
            adam_beta2: float = 0.999,
            adam_eps: float = 1e-08,
            save_dir: str | os.PathLike = None,
            temp_dir: str | os.PathLike = None,
    ) -> None:
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.temp_dir = temp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create the dataset
        self.dataset = build_molecule_dataset(
            dataset_type='denoise',
            img_dir=img_dir,
            psf_dir=psf_dir,
            train=True
        )
        self.dataloader = build_dataloader(
            dataset=self.dataset,
            batches_per_gpu=num_batches,
            workers_per_gpu=num_workers,
            num_gpus=1,
            shuffle=True,
            seed=manual_seed,
            pin_memory=False
        )

        # create the Generator and apply the `init_weight` function to randomly initialize all weights
        # create the Discriminator and apply the `init_weight` function to randomly initialize all weights
        self.net_g = UNetGenerator().to(self.device)
        self.net_g.apply(init_weights)
        self.net_d = CNNDiscriminator().to(self.device)
        self.net_d.apply(init_weights)

        # initialize the loss function
        self.criterion_g = GeneratorLoss(l1_weight=5, device='{}:{}'.format(self.device.type, self.device.index))
        self.criterion_d = DiscriminatorLoss(device='{}:{}'.format(self.device.type, self.device.index))

        # initialize the optimizers
        self.optimizer_g = torch.optim.Adam(
            self.net_g.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps
        )
        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps
        )

        # initialize the learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_g,
            1.0,
            0.1,
            num_epochs - 1
        )
        self.scheduler_d = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_d,
            1.0,
            0.1,
            num_epochs - 1
        )

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
            from tqdm import tqdm as progressbar
        return progressbar

    def train(self) -> Tuple[List, List]:
        """Train GAN.
        """

        # tensorboard summary writer
        summary_writer = SummaryWriter(os.path.join('runs', 'cfgan'))
        logger.info('Running `tensorboard --logdir=runs` from command line.')

        summary_writer.add_graph(self.net_g, torch.randn(1, 1, 256, 256).to(self.device))
        summary_writer.add_graph(self.net_d, torch.randn(1, 1, 256, 256).to(self.device))

        # lists to keep track of progress
        g_losses = []
        d_losses = []

        iters = 0

        # start training loop
        logger.info('Starting training loop on %s.', self.device)

        progressbar = self._set_progressbar()

        # for each epoch
        for epoch in progressbar(range(self.num_epochs), 'Training...'):
            # for each batch in the dataloader
            for i, data in enumerate(self.dataloader):  # type: int, Tuple[torch.Tensor, torch.Tensor]
                # ``noise`` is the cross-filter image, ``clear`` is the corresponding psf
                noise = data[0]
                clear = data[1]
                real_image: torch.Tensor = clear.to(self.device)
                fake_image: torch.Tensor = self.net_g(noise.to(self.device))

                # update Discriminator with real image (psf) => ``D(y)``
                # and fake image (generated cross-filter image) => ``D(G(x))``
                self.net_d.zero_grad()
                output_d_real = self.net_d(real_image)
                output_d_fake = self.net_d(fake_image.detach())
                err_d = self.criterion_d(output_d_real, output_d_fake)
                err_d.backward(retain_graph=True)
                d_y = output_d_real.mean().item()
                self.optimizer_d.step()

                # update Generator with fake image (generated cross-filter image)  => ``G(x)``
                self.net_g.zero_grad()
                output_g = self.net_d(fake_image)
                err_g = self.criterion_g(fake_image, real_image) + (output_g.mean() - 1) ** 2
                err_g.backward()
                d_g_x1 = output_d_fake.mean().item()
                d_g_x2 = output_g.mean().item()
                self.optimizer_g.step()

                # output training states
                # check the performance of Generator
                if (i + 1) % 10 == 0:
                    # monitor disk usage
                    disk_usage = monitor_disk_usage()
                    gpu_free_memory = monitor_gpu_memory(self.device.index)

                    print(
                        '[{:3d}/{:3d}][{:3d}/{:3d}]\t'
                        'Loss_D: {:.4f}, Loss_G: {:.4f}  '
                        'D(y): {:.4f}, D(G(x)): {:.4f} / {:.4f}    '
                        'Disk usage: {}    '
                        'GPU free memory: {:.4f} MB'.format(
                            epoch + 1, self.num_epochs, i + 1, len(self.dataloader),
                            err_d.item(), err_g.item(),
                            d_y, d_g_x1, d_g_x2,
                            disk_usage,
                            gpu_free_memory
                        )
                    )
                    summary_writer.add_scalars(
                        'training/loss',
                        {
                            'Loss_D': err_d.item(),
                            'Loss_G': err_g.item()
                        },
                        iters
                    )
                    summary_writer.add_scalars(
                        'training/dis',
                        {
                            'D(y)': d_y,
                            'D(G(x))_1': d_g_x1,
                            'D(G(x))_2': d_g_x2
                        },
                        iters
                    )
                    summary_writer.add_scalars(
                        'training/disk_usage',
                        disk_usage,
                        iters
                    )
                    summary_writer.add_scalars(
                        'training/gpu_free_memory',
                        {'gpu_free_memory': gpu_free_memory},
                        iters
                    )
                    summary_writer.add_scalars(
                        'training/learning_rate',
                        {
                            'lr_g': self.optimizer_g.state_dict()['param_groups'][0]['lr'],
                            'lr_d': self.optimizer_d.state_dict()['param_groups'][0]['lr']
                        },
                        iters
                    )

                # save losses for plotting
                g_losses.append(err_g.item())
                d_losses.append(err_d.item())

                iters += 1

            # update learning rate schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()

            # save temporary checkpoints
            if (epoch + 1) % 50 == 0 and self.temp_dir:
                torch.save(self.net_g, os.path.join(f'{self.temp_dir}', f'gen_epoch{epoch + 1}.pt'))
                torch.save(self.net_d, os.path.join(f'{self.temp_dir}', f'dis_epoch{epoch + 1}.pt'))

        # finish training loop
        summary_writer.close()
        logger.info('Finished training loop.')

        # save checkpoints
        if self.save_dir:
            torch.save(self.net_g, os.path.join(f'{self.save_dir}', 'gen.pt'))
            torch.save(self.net_d, os.path.join(f'{self.save_dir}', 'dis.pt'))
            logger.info('Trained checkpoint saved.')

        return g_losses, d_losses


if __name__ == '__main__':
    config_file = 'config.ini'
    dataset_config = CFGANDatasetConfig.from_ini_file(config_file)
    trainer_config = TrainerConfig.from_ini_file(config_file)

    trainer = CFGANTrainer(
        dataset_config.img_dir,
        dataset_config.psf_dir,
        trainer_config.num_epochs,
        trainer_config.num_batches,
        trainer_config.num_workers,
        trainer_config.learning_rate,
        trainer_config.adam_beta1,
        trainer_config.adam_beta2,
        trainer_config.adam_eps,
        trainer_config.save_dir,
        trainer_config.temp_dir
    )

    g_losses, d_losses = trainer.train()
    np.save(os.path.join('checkpoints', 'g_losses.npy'), g_losses)
    np.save(os.path.join('checkpoints', 'd_losses.npy'), d_losses)
