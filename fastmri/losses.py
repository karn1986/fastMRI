"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
    
class SelfSupervisedLoss(nn.Module):
    def __init__(self, 
                 # dc_weight: float = 1e3, 
                 # energy_weight: float= 100,
                 intensity_weight: float = 20,
                 hist_weight: float=10,
                 bins: int=5,
                 tv_weight: float = 1e5):
        """
        Args:
            dc_weight: Weight for data consistency loss.
            tv_weight: Weight for total variation loss.
        """
        super().__init__()
        # self.dc_weight = dc_weight
        # self.energy_weight = energy_weight
        self.intensity_weight = intensity_weight
        self.hist_weight = hist_weight
        self.bins = bins
        self.tv_weight = tv_weight
    
    @staticmethod
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
   
    def forward(self, 
                kspace_pred: torch.Tensor, 
                ref_kspace: torch.Tensor):
        """
            Compute data consistency loss in kspace and 
            total variation loss in image space.
    
            Inputs:
            - kspace_pred: PyTorch tensor of shape (N, H, W, 2) holding predicted kspace.
            - ref_kspace : input masked kspace 
            - mask: the subsampling mask
    
            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img.
        """
        # zero = torch.zeros(1, 1, 1, 1).to(kspace_pred)
        # dc_loss = torch.where(mask, kspace_pred - ref_kspace, zero)
        # dc_loss = torch.norm(dc_loss)
        # dc_loss = torch.max(torch.abs(dc_loss))
        # dc_loss = dc_loss / torch.max(torch.abs(ref_kspace))
                
        output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
        gt     = fastmri.complex_abs(fastmri.ifft2c(ref_kspace))
        # energy_loss = torch.abs(torch.sum(output) - torch.sum(gt))
        
        hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
        hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
        hmin_pred, hmax_pred = hdiff_pred.min().item(), hdiff_pred.max().item()
        hmin_gt, hmax_gt = hdiff_gt.min().item(), hdiff_gt.max().item()
        hist_x = self.differentiable_histogram(hdiff_pred, bins = self.bins, min = hmin_pred, max = hmax_pred)
        hist_y = self.differentiable_histogram(hdiff_gt, bins = self.bins, min = hmin_gt, max = hmax_gt)
        hdiff_hist_loss = (hist_x-hist_y)/len(hdiff_pred)
        hdiff_hist_loss = torch.norm(hdiff_hist_loss)
 
        vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
        vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)
        vmin_pred, vmax_pred = vdiff_pred.min().item(), vdiff_pred.max().item()
        vmin_gt, vmax_gt = vdiff_gt.min().item(), vdiff_gt.max().item()
        hist_x = self.differentiable_histogram(vdiff_pred, bins = self.bins, min = vmin_pred, max = vmax_pred)
        hist_y = self.differentiable_histogram(vdiff_gt, bins = self.bins, min = vmin_gt, max = vmax_gt)
        vdiff_hist_loss = (hist_x-hist_y)/len(vdiff_pred)
        vdiff_hist_loss = torch.norm(vdiff_hist_loss)
        
        output = output.view(-1)
        gt = gt.view(-1)
        gt_min, gt_max = gt.min().item(), gt.max().item()
        hist_x = self.differentiable_histogram(output, bins = self.bins, min = gt_min, max = gt_max)
        hist_y = self.differentiable_histogram(gt, bins = self.bins, min = gt_min, max = gt_max)
        gt_hist_loss = (hist_x-hist_y)/len(output)
        gt_hist_loss = torch.norm(gt_hist_loss)

        hdiff_var_loss = torch.sqrt(torch.var(hdiff_pred)) - 1.5 * torch.sqrt(torch.var(hdiff_gt))
        vdiff_var_loss = torch.sqrt(torch.var(vdiff_pred)) - 1.5 * torch.sqrt(torch.var(vdiff_gt))
        # intensity_var_loss = torch.abs(torch.sqrt(torch.var(output.view(-1))) - 
        #                       torch.sqrt(torch.var(gt.view(-1))))

        tv_loss = (torch.abs(hdiff_var_loss) + torch.abs(vdiff_var_loss))
        hist_loss = hdiff_hist_loss + vdiff_hist_loss
        return (self.intensity_weight * gt_hist_loss + 
                self.hist_weight*hist_loss + self.tv_weight*tv_loss)   
# class SelfSupervisedLoss(nn.Module):
#     def __init__(self, 
#                  # dc_weight: float = 1e3, 
#                  # energy_weight: float= 100,
#                  intensity_weight: float = 20,
#                  hist_weight: float=10,
#                  bins: int=5,
#                  tv_weight: float = 1e5):
#         """
#         Args:
#             dc_weight: Weight for data consistency loss.
#             tv_weight: Weight for total variation loss.
#         """
#         super().__init__()
#         # self.dc_weight = dc_weight
#         # self.energy_weight = energy_weight
#         self.intensity_weight = intensity_weight
#         self.hist_weight = hist_weight
#         self.bins = bins
#         self.tv_weight = tv_weight
        
#     def forward(self, 
#                 kspace_pred: torch.Tensor, 
#                 ref_kspace: torch.Tensor,
#                 mask: torch.Tensor):
#         """
#             Compute data consistency loss in kspace and 
#             total variation loss in image space.
    
#             Inputs:
#             - kspace_pred: PyTorch tensor of shape (N, H, W, 2) holding predicted kspace.
#             - ref_kspace : input masked kspace 
#             - mask: the subsampling mask
    
#             Returns:
#             - loss: PyTorch Variable holding a scalar giving the total variation loss
#               for img.
#         """
#         # zero = torch.zeros(1, 1, 1, 1).to(kspace_pred)
#         # dc_loss = torch.where(mask, kspace_pred - ref_kspace, zero)
#         # dc_loss = torch.norm(dc_loss)
#         # dc_loss = torch.max(torch.abs(dc_loss))
#         # dc_loss = dc_loss / torch.max(torch.abs(ref_kspace))
                
#         output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
#         gt     = fastmri.complex_abs(fastmri.ifft2c(ref_kspace))
#         # energy_loss = torch.abs(torch.sum(output) - torch.sum(gt))
#         zero = torch.zeros(1).to(output)
#         one  = torch.ones(1).to(output)
        
#         hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
#         hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
#         hmin_pred, hmax_pred = hdiff_pred.min(), hdiff_pred.max()
#         hmin_gt, hmax_gt = hdiff_gt.min(), hdiff_gt.max()
#         hstep_pred = (hmax_pred - hmin_pred)/self.bins
#         hstep_gt = (hmax_gt - hmin_gt)/self.bins
#         hdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#         hones = hdiff_pred/hdiff_pred
 
#         vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
#         vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)
#         vmin_pred, vmax_pred = vdiff_pred.min(), vdiff_pred.max()
#         vmin_gt, vmax_gt = vdiff_gt.min(), vdiff_gt.max()
#         vstep_pred = (vmax_pred - vmin_pred)/self.bins
#         vstep_gt = (vmax_gt - vmin_gt)/self.bins
#         vdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#         vones = vdiff_pred/vdiff_pred
        
#         output = output.view(-1)
#         gt = gt.view(-1)
#         gt_min, gt_max = gt.min(), gt.max()
#         step_gt = (gt_max - gt_min)/self.bins
#         gt_hist_loss = torch.zeros(1, requires_grad=True).to(output)
#         nh, nv, n = len(hdiff_pred), len(vdiff_pred), len(gt)
#         ones = output/output
#         for i in range(self.bins):
#             # x is the empirical distribution of the predictions
#             # y is the empirical distribution of the ground truth
#             x = torch.sum(torch.where(((hdiff_pred>hmin_pred+i*hstep_pred) & 
#                                       (hdiff_pred<hmin_pred+(i+1)*hstep_pred)), hones, zero))
#             y = torch.sum(torch.where(((hdiff_gt>hmin_gt+i*hstep_gt) & 
#                                       (hdiff_gt<hmin_gt+(i+1)*hstep_gt)), one, zero))
#             hdiff_hist_loss = hdiff_hist_loss + ((x-y)/nh)**2
#             x = torch.sum(torch.where(((vdiff_pred>vmin_pred+i*vstep_pred) & 
#                                       (vdiff_pred<vmin_pred+(i+1)*vstep_pred)), vones, zero))
#             y = torch.sum(torch.where(((vdiff_gt>vmin_gt+i*vstep_gt) & 
#                                       (vdiff_gt<vmin_gt+(i+1)*vstep_gt)), one, zero))
#             vdiff_hist_loss = vdiff_hist_loss + ((x-y)/nv)**2
#             x = torch.sum(torch.where(((output>gt_min+i*step_gt) & 
#                                       (output<gt_min+(i+1)*step_gt)), ones, zero))
#             y = torch.sum(torch.where(((gt>gt_min+i*step_gt) & 
#                                       (gt<gt_min+(i+1)*step_gt)), one, zero))
#             gt_hist_loss = gt_hist_loss + ((x-y)/n)**2
#         hdiff_hist_loss = torch.sqrt(hdiff_hist_loss)
#         vdiff_hist_loss = torch.sqrt(vdiff_hist_loss)
#         gt_hist_loss    = torch.sqrt(gt_hist_loss)

#         hdiff_var_loss = torch.sqrt(torch.var(hdiff_pred)) - 1.5 * torch.sqrt(torch.var(hdiff_gt))
#         vdiff_var_loss = torch.sqrt(torch.var(vdiff_pred)) - 1.5 * torch.sqrt(torch.var(vdiff_gt))
#         # intensity_var_loss = torch.abs(torch.sqrt(torch.var(output.view(-1))) - 
#         #                       torch.sqrt(torch.var(gt.view(-1))))

#         tv_loss = (torch.abs(hdiff_var_loss) + torch.abs(vdiff_var_loss))
#         hist_loss = hdiff_hist_loss + vdiff_hist_loss
#         return (self.intensity_weight * gt_hist_loss + 
#                 self.hist_weight*hist_loss + self.tv_weight*tv_loss)

class TotalVariationLoss(nn.Module):
    def __init__(self, tv_weight: float = 1e5):
        """
        Args:
            tv_weight: Weight for total variation loss.
        """
        super().__init__()
        self.tv_weight = tv_weight
        
    def forward(self, 
                kspace_pred: torch.Tensor, 
                ref_kspace: torch.Tensor):
        """
            Compute data consistency loss in kspace and 
            total variation loss in image space.
    
            Inputs:
            - kspace_pred: PyTorch tensor of shape (N, H, W, 2) holding predicted kspace.
            - ref_kspace : input masked kspace 
    
            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img.
        """
                
        output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
        gt     = fastmri.complex_abs(fastmri.ifft2c(ref_kspace))
        # energy_loss = torch.abs(torch.sum(output) - torch.sum(gt))

        hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
        hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
 
        vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
        vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)

        
      
        hdiff_var_loss = torch.sqrt(torch.var(hdiff_pred)) - 1.25 * torch.sqrt(torch.var(hdiff_gt))
        vdiff_var_loss = torch.sqrt(torch.var(vdiff_pred)) - 1.25 * torch.sqrt(torch.var(vdiff_gt))

        tv_loss = (torch.abs(hdiff_var_loss) + torch.abs(vdiff_var_loss))
        return self.tv_weight*tv_loss