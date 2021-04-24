"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from .unet import Unet
#%%
# def hist_loss(current_kspace: torch.Tensor, 
#               masked_kspace: torch.Tensor,
#               bins: int=5):
#     """    
#         Inputs:
#         - kspace_pred: PyTorch tensor of shape (N, H, W, 2) holding predicted kspace.
#         - ref_kspace : input masked kspace 
#         - mask: the subsampling mask

#         Returns:
#         - loss: PyTorch Variable holding a scalar giving the total variation loss
#           for img.
#     """
#     output = fastmri.complex_abs(fastmri.ifft2c(current_kspace))
#     gt     = fastmri.complex_abs(fastmri.ifft2c(masked_kspace))
    
#     zero = torch.zeros(1).to(output)
#     one  = torch.ones(1).to(output)
        
#     hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
#     hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
#     hmin_pred, hmax_pred = hdiff_pred.min(), hdiff_pred.max()
#     hmin_gt, hmax_gt = hdiff_gt.min(), hdiff_gt.max()
#     hstep_pred = (hmax_pred - hmin_pred)/bins
#     hstep_gt = (hmax_gt - hmin_gt)/bins
#     hdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#     hones = hdiff_pred/hdiff_pred
 
#     vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
#     vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)
#     vmin_pred, vmax_pred = vdiff_pred.min(), vdiff_pred.max()
#     vmin_gt, vmax_gt = vdiff_gt.min(), vdiff_gt.max()
#     vstep_pred = (vmax_pred - vmin_pred)/bins
#     vstep_gt = (vmax_gt - vmin_gt)/bins
#     vdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#     vones = vdiff_pred/vdiff_pred
    
#     output = output.view(-1)
#     gt = gt.view(-1)
#     gt_min, gt_max = gt.min(), gt.max()
#     step_gt = (gt_max - gt_min)/bins
#     gt_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#     ones = output/output
#     nh, nv, n = len(hdiff_pred), len(vdiff_pred), len(gt)
#     for i in range(bins):
#         # x is the empirical distribution of the predictions
#         # y is the empirical distribution of the ground truth
#         x = torch.sum(torch.where(((hdiff_pred>hmin_pred+i*hstep_pred) & 
#                                   (hdiff_pred<hmin_pred+(i+1)*hstep_pred)), hones, zero))
#         y = torch.sum(torch.where(((hdiff_gt>hmin_gt+i*hstep_gt) & 
#                                   (hdiff_gt<hmin_gt+(i+1)*hstep_gt)), one, zero))
#         hdiff_hist_loss = hdiff_hist_loss + ((x-y)/nh)**2
#         x = torch.sum(torch.where(((vdiff_pred>vmin_pred+i*vstep_pred) & 
#                                   (vdiff_pred<vmin_pred+(i+1)*vstep_pred)), vones, zero))
#         y = torch.sum(torch.where(((vdiff_gt>vmin_gt+i*vstep_gt) & 
#                                   (vdiff_gt<vmin_gt+(i+1)*vstep_gt)), one, zero))
#         vdiff_hist_loss = vdiff_hist_loss + ((x-y)/nv)**2
#         x = torch.sum(torch.where(((output>gt_min+i*step_gt) & 
#                                   (output<gt_min+(i+1)*step_gt)), ones, zero))
#         y = torch.sum(torch.where(((gt>gt_min+i*step_gt) & 
#                                   (gt<gt_min+(i+1)*step_gt)), one, zero))
#         gt_hist_loss = gt_hist_loss + ((x-y)/n)**2
#     hdiff_hist_loss = torch.sqrt(hdiff_hist_loss)
#     vdiff_hist_loss = torch.sqrt(vdiff_hist_loss)
#     gt_hist_loss    = torch.sqrt(gt_hist_loss)

#     return (hdiff_hist_loss + vdiff_hist_loss + gt_hist_loss)
def differentiable_histogram(x, bins=100, min=0.0, max=1.0):
    hist_torch = torch.zeros(bins).to(x.device)
    delta = (max - min) / bins

    for i in range(bins):
        h_l = min + i * delta
        h_u = min + (i + 1) * delta

        mask = ((h_u > x) & (x >= h_l)).float()

        hist_torch[i] += torch.sum((x - h_l) * mask)
        hist_torch[i] += torch.sum((h_u - x) * mask)

    return hist_torch / delta
    
def hist_loss(current_kspace: torch.Tensor, 
              masked_kspace: torch.Tensor,
              bins: int=5):
    """    
        Inputs:
        - kspace_pred: PyTorch tensor of shape (N, H, W, 2) holding predicted kspace.
        - ref_kspace : input masked kspace 
        - mask: the subsampling mask
    
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img.
    """
    output = fastmri.complex_abs(fastmri.ifft2c(current_kspace))
    gt     = fastmri.complex_abs(fastmri.ifft2c(masked_kspace))
    
    hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
    hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
    hmin_pred, hmax_pred = hdiff_pred.min().item(), hdiff_pred.max().item()
    hmin_gt, hmax_gt = hdiff_gt.min().item(), hdiff_gt.max().item()
    hist_x = differentiable_histogram(hdiff_pred, bins = bins, min = hmin_pred, max = hmax_pred)
    hist_y = differentiable_histogram(hdiff_gt, bins = bins, min = hmin_gt, max = hmax_gt)
    hdiff_hist_loss = (hist_x-hist_y)/len(hdiff_pred)
    hdiff_hist_loss = torch.norm(hdiff_hist_loss)
     
    vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
    vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)
    vmin_pred, vmax_pred = vdiff_pred.min().item(), vdiff_pred.max().item()
    vmin_gt, vmax_gt = vdiff_gt.min().item(), vdiff_gt.max().item()
    hist_x = differentiable_histogram(vdiff_pred, bins = bins, min = vmin_pred, max = vmax_pred)
    hist_y = differentiable_histogram(vdiff_gt, bins = bins, min = vmin_gt, max = vmax_gt)
    vdiff_hist_loss = (hist_x-hist_y)/len(vdiff_pred)
    vdiff_hist_loss = torch.norm(vdiff_hist_loss)
    
    output = output.view(-1)
    gt = gt.view(-1)
    gt_min, gt_max = gt.min().item(), gt.max().item()
    hist_x = differentiable_histogram(output, bins = bins, min = gt_min, max = gt_max)
    hist_y = differentiable_histogram(gt, bins = bins, min = gt_min, max = gt_max)
    gt_hist_loss = (hist_x-hist_y)/len(output)
    gt_hist_loss = torch.norm(gt_hist_loss)
    
    return (hdiff_hist_loss + vdiff_hist_loss + gt_hist_loss)

#%%
class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 3, 1, 2)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, two, h, w = x.shape
        assert two == 2
        return x.permute(0, 2, 3, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class ssVarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask)

        return kspace_pred


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.hist_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        with torch.enable_grad():
            X = current_kspace.clone().detach().requires_grad_(True)
            histl = hist_loss(X, ref_kspace.clone().detach())
            histl.backward()
            soft_histc = X.grad * self.hist_weight

        model_term = fastmri.fft2c(self.model(fastmri.ifft2c(current_kspace)))

        return current_kspace - soft_dc - soft_histc - model_term
