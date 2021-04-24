"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from fastmri.models import ssVarNet as VarNet

from .mri_module import MriModule


class SSVarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SelfSupervisedLoss()

    def forward(self, masked_kspace, mask):
        return self.varnet(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, _, _, _, max_value, _ = batch

        kspace_pred = self(masked_kspace, mask)
        # target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(kspace_pred, masked_kspace)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch

        kspace_pred = self(masked_kspace, mask)
        output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(kspace_pred, masked_kspace),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet
        
        kspace_pred = self(masked_kspace, mask)
        output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
