#!/usr/bin/env python
# coding: utf-8

# #### This notebook shows how to read the fastMRI dataset and apply some simple transformations to the data.

# In[1]:


# Testing if integration works


# In[2]:

import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
import matplotlib.pyplot as plt
plt.close('all')

# The fastMRI dataset is distributed as a set of HDF5 files and can be read with the h5py package. Here, we show how to open a file from the multi-coil dataset. Each file corresponds to one MRI scan and contains the k-space data, ground truth and some meta data related to the scan.

# In[3]:


file_name = 'singlecoil_val/file1002280.h5'
hf = h5py.File(file_name)


# In[4]:


print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))


# In multi-coil MRIs, k-space has the following shape:
# (number of slices, number of coils, height, width)
# 
# For single-coil MRIs, k-space has the following shape:
# (number of slices, height, width)
# 
# MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices.

# In[5]:


volume_kspace = hf['kspace'][()]
volume_kspace = volume_kspace[:,np.newaxis,:,:]
print(volume_kspace.dtype)
print(volume_kspace.shape)


# In[6]:


slice_kspace = volume_kspace[20] # Choosing the 20-th slice of this volume


# Let's see what the absolute value of k-space looks like:

# In[7]:


def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)


# In[8]:


show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0])  # This shows coils 0, 5 and 10


# The fastMRI repo contains some utlity functions to convert k-space into image space. These functions work on PyTorch Tensors. The to_tensor function can convert Numpy arrays to PyTorch Tensors.

# In[9]:


import fastmri
from fastmri.data import transforms as T


# In[10]:


slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

# SSIM loss
loss = fastmri.SSIMLoss()
print(loss(slice_image_abs.unsqueeze(1), slice_image_abs.unsqueeze(1), data_range=slice_image_abs.max().reshape(-1)))
# In[15]:


show_coils(slice_image_abs, [0], cmap='gray')


# As we can see, each coil in a multi-coil MRI scan focusses on a different region of the image. These coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

# In[16]:


slice_image_rss = fastmri.rss(slice_image_abs, dim=0)


# In[17]:


plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')


# So far, we have been looking at fully-sampled data. We can simulate under-sampled data by creating a mask and applying it to k-space.

# In[18]:


from fastmri.data.subsample import RandomMaskFunc
mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])  # Create the mask function object


# In[19]:


masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space


# Let's see what the subsampled image looks like:

# In[20]:


sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
show_coils(sampled_image_abs, [0], cmap='gray')

ckpt_path = "fastmri_examples/varnet/varnet/varnet_demo/checkpoints/epoch=6-step=72960.ckpt"
from fastmri.pl_modules import SSVarNetModule as VarNetModule
model = VarNetModule.load_from_checkpoint(ckpt_path)
kspace_pred = model(masked_kspace, mask.byte())
# In[ ]:
print(torch.max(masked_kspace))
print(torch.min(masked_kspace))
print(torch.max(torch.abs(masked_kspace)))
print(torch.min(torch.abs(masked_kspace)))

output = fastmri.complex_abs(fastmri.ifft2c(kspace_pred))
show_coils(output.detach().numpy(), [0], cmap='gray')

plt.style.use('seaborn-white')
kwargs = dict(histtype='stepfilled', alpha=0.3, bins=100)
fig, ax = plt.subplots()
ax.hist(output.view(-1).detach().numpy(), label='Reconstructed', **kwargs)
ax.hist(sampled_image_abs.view(-1).numpy(), label = 'Aliased', **kwargs)
ax.hist(slice_image_abs.view(-1).numpy(), label = 'Ground Truth', **kwargs)
ax.legend()

hdiff = (output[:,:,:-1] - output[:,:,1:])
vdiff = (output[:,:-1,:] - output[:,1:,:])
_, ax1 = plt.subplots()
_, ax2 = plt.subplots()
ax1.hist(hdiff.view(-1).detach().numpy(),label = 'Reconstructed', **kwargs)
ax2.hist(vdiff.view(-1).detach().numpy(), label= 'Reconstructed', **kwargs)

hdiff = (sampled_image_abs[:,:,:-1] - sampled_image_abs[:,:,1:])
vdiff = (sampled_image_abs[:,:-1,:] - sampled_image_abs[:,1:,:])

ax1.hist(hdiff.view(-1).numpy(), label = 'Aliased', **kwargs)
ax2.hist(vdiff.view(-1).numpy(), label = 'Aliased', **kwargs)

hdiff = (slice_image_abs[:,:,:-1] - slice_image_abs[:,:,1:])
vdiff = (slice_image_abs[:,:-1,:] - slice_image_abs[:,1:,:])

ax1.hist(hdiff.view(-1).numpy(), label = 'Ground truth', **kwargs)
ax2.hist(vdiff.view(-1).numpy(), label = 'Ground truth', **kwargs)
ax1.legend()
ax2.legend()
# In[ ]:
hdiff = (output[:,:,:-1] - output[:,:,1:])
vdiff = (output[:,:-1,:] - output[:,1:,:])
_, ax1 = plt.subplots()
_, ax2 = plt.subplots()
kwargs = dict(alpha=0.3, step = 'mid')
hdiff_hist = torch.histc(hdiff.view(-1))/len(hdiff.view(-1))
vdiff_hist = torch.histc(vdiff.view(-1))/len(vdiff.view(-1))
# x = np.linspace(hdiff.detach().numpy().min(), hdiff.detach().numpy().max(), 100, endpoint=False)
# ax1.fill_between(x, hdiff_hist.detach().numpy() ,label = 'Reconstructed', **kwargs)
# x = np.linspace(vdiff.detach().numpy().min(), vdiff.detach().numpy().max(), 100, endpoint=False)
# ax2.fill_between(x, vdiff_hist.detach().numpy() ,label = 'Reconstructed', **kwargs)

hdiff = (sampled_image_abs[:,:,:-1] - sampled_image_abs[:,:,1:])
vdiff = (sampled_image_abs[:,:-1,:] - sampled_image_abs[:,1:,:])

hdiff_hist = torch.histc(hdiff.view(-1))/len(hdiff.view(-1))
vdiff_hist = torch.histc(vdiff.view(-1))/len(vdiff.view(-1))
x = np.linspace(hdiff.numpy().min(), hdiff.numpy().max(), 100, endpoint=False)
ax1.fill_between(2*x, hdiff_hist.numpy() ,label = 'Aliased', **kwargs)
x = np.linspace(vdiff.numpy().min(), vdiff.numpy().max(), 100, endpoint=False)
ax2.fill_between(2*x, vdiff_hist.numpy() ,label = 'Aliased', **kwargs)

hdiff = (slice_image_abs[:,:,:-1] - slice_image_abs[:,:,1:])
vdiff = (slice_image_abs[:,:-1,:] - slice_image_abs[:,1:,:])

hdiff_hist = torch.histc(hdiff.view(-1))/len(hdiff.view(-1))
vdiff_hist = torch.histc(vdiff.view(-1))/len(vdiff.view(-1))
x = np.linspace(hdiff.numpy().min(), hdiff.numpy().max(), 100, endpoint=False)
ax1.fill_between(x, hdiff_hist.numpy() ,label = 'Ground Truth', **kwargs)
x = np.linspace(vdiff.numpy().min(), vdiff.numpy().max(), 100, endpoint=False)
ax2.fill_between(x, vdiff_hist.numpy() ,label = 'Ground Truth', **kwargs)
ax1.legend()
ax2.legend()
#%% Compare manual histogram calculation with torch.histc
gt  = sampled_image_abs
bins = 100
hdiff_pred = (output[:,:,:-1] - output[:,:,1:]).view(-1)
hdiff_gt   = (gt[:,:,:-1] - gt[:,:,1:]).view(-1)
hmin_pred, hmax_pred = hdiff_pred.min(), hdiff_pred.max()
hmin_gt, hmax_gt = hdiff_gt.min(), hdiff_gt.max()
hstep_pred = (hmax_pred - hmin_pred)/bins
hstep_gt = (hmax_gt - hmin_gt)/bins
hdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
zero = torch.zeros(1, requires_grad=True).to(output)
one = torch.ones(1, requires_grad=True).to(output)
 
vdiff_pred = (output[:,:-1,:] - output[:,1:,:]).view(-1)
vdiff_gt   = (gt[:,:-1,:] - gt[:,1:,:]).view(-1)
vmin_pred, vmax_pred = vdiff_pred.min(), vdiff_pred.max()
vmin_gt, vmax_gt = vdiff_gt.min(), vdiff_gt.max()
vstep_pred = (vmax_pred - vmin_pred)/bins
vstep_gt = (vmax_gt - vmin_gt)/bins
vdiff_hist_loss = torch.zeros(1, requires_grad=True).to(output)
output = output.view(-1)
gt = gt.view(-1)
gt_min, gt_max = gt.min(), gt.max()
step_gt = (gt_max - gt_min)/bins
gt_hist_loss = torch.zeros(1, requires_grad=True).to(output)
pred_prob = torch.zeros(1, requires_grad=True).to(output)
gt_prob = torch.zeros(1, requires_grad=True).to(output)
nh, nv, n = len(hdiff_pred), len(vdiff_pred), len(gt)

hdiff_hist_gt = torch.empty(bins)
hdiff_hist_pred = torch.empty(bins)
vdiff_hist_gt = torch.empty(bins)
vdiff_hist_pred = torch.empty(bins)
for i in range(bins):
    x = torch.sum(torch.where(((hdiff_pred>hmin_pred+i*hstep_pred) & 
                              (hdiff_pred<hmin_pred+(i+1)*hstep_pred)), one, zero))
    y = torch.sum(torch.where(((hdiff_gt>hmin_gt+i*hstep_gt) & 
                              (hdiff_gt<hmin_gt+(i+1)*hstep_gt)), one, zero))
    hdiff_hist_pred[i] = x/nh
    hdiff_hist_gt[i] = y/nh
    hdiff_hist_loss = hdiff_hist_loss + ((x-y)/nh)**2
    x = torch.sum(torch.where(((vdiff_pred>vmin_pred+i*vstep_pred) & 
                              (vdiff_pred<vmin_pred+(i+1)*vstep_pred)), one, zero))
    y = torch.sum(torch.where(((vdiff_gt>vmin_gt+i*vstep_gt) & 
                              (vdiff_gt<vmin_gt+(i+1)*vstep_gt)), one, zero))
    vdiff_hist_pred[i] = x/nv
    vdiff_hist_gt[i] = y/nv
    vdiff_hist_loss = vdiff_hist_loss + ((x-y)/nv)**2
    x = torch.sum(torch.where(((output>gt_min+i*step_gt) & 
                              (output<gt_min+(i+1)*step_gt)), one, zero))
    y = torch.sum(torch.where(((gt>gt_min+i*step_gt) & 
                              (gt<gt_min+(i+1)*step_gt)), one, zero))
    gt_hist_loss = gt_hist_loss + ((x-y)/n)**2
    pred_prob = pred_prob + x/n
    gt_prob = gt_prob + y/n
hdiff_hist_loss = torch.sqrt(hdiff_hist_loss)
vdiff_hist_loss = torch.sqrt(vdiff_hist_loss)
gt_hist_loss = torch.sqrt(gt_hist_loss)
hdiff_var_loss = torch.sqrt(torch.var(hdiff_pred)) - 2 * torch.sqrt(torch.var(hdiff_gt))
vdiff_var_loss = torch.sqrt(torch.var(vdiff_pred)) - 2 * torch.sqrt(torch.var(vdiff_gt)) 

tv_loss = torch.abs(hdiff_var_loss) + torch.abs(vdiff_var_loss)
hist_loss = hdiff_hist_loss + vdiff_hist_loss
# _, ax1 = plt.subplots()
# _, ax2 = plt.subplots()
# kwargs = dict(alpha=0.3, step = 'mid')

# hdiff = (output[:,:,:-1] - output[:,:,1:])
# vdiff = (output[:,:-1,:] - output[:,1:,:])

# hdiff_hist = torch.histc(hdiff.view(-1))/len(hdiff.view(-1))
# vdiff_hist = torch.histc(vdiff.view(-1))/len(vdiff.view(-1))
# x = np.linspace(hmin_pred.detach().numpy(), hmax_pred.detach().numpy(), 100, endpoint=False)
# ax1.fill_between(x, hdiff_hist_pred.detach().numpy() ,label = 'MANUAL RECON', **kwargs)
# ax1.fill_between(x, hdiff_hist.detach().numpy() ,label = 'HISTC RECON', **kwargs)
# x = np.linspace(vmin_pred.detach().numpy(), vmax_pred.detach().numpy(), 100, endpoint=False)
# ax2.fill_between(x, vdiff_hist_pred.detach().numpy() ,label = 'MANUAL RECON', **kwargs)
# ax2.fill_between(x, vdiff_hist_pred.detach().numpy() ,label = 'HISTC RECON', **kwargs)
# ax1.legend()
# ax2.legend()

# _, ax1 = plt.subplots()
# _, ax2 = plt.subplots()
# kwargs = dict(alpha=0.3, step = 'mid')
# hdiff = (sampled_image_abs[:,:,:-1] - sampled_image_abs[:,:,1:])
# vdiff = (sampled_image_abs[:,:-1,:] - sampled_image_abs[:,1:,:])

# hdiff_hist = torch.histc(hdiff.view(-1))/len(hdiff.view(-1))
# vdiff_hist = torch.histc(vdiff.view(-1))/len(vdiff.view(-1))
# x = np.linspace(hmin_gt.numpy(), hmax_gt.numpy(), 100, endpoint=False)
# ax1.fill_between(x, hdiff_hist_gt.detach().numpy() ,label = 'MANUAL ALIASED', **kwargs)
# ax1.fill_between(x, hdiff_hist.detach().numpy() ,label = 'TORCH HIST ALIASED', **kwargs)
# x = np.linspace(vmin_gt.numpy(), vmax_gt.numpy(), 100, endpoint=False)
# ax2.fill_between(x, vdiff_hist_gt.detach().numpy() ,label = 'MANUAL ALIASED', **kwargs)
# ax2.fill_between(x, vdiff_hist.detach().numpy() ,label = 'TORCHHIST ALIASED', **kwargs)
# ax1.legend()
# ax2.legend()