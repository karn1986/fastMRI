# -*- coding: utf-8 -*-
"""
Dataloader example
"""

import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = transforms.to_tensor(kspace)
    masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
    return masked_kspace, mask, target

dataset = mri_data.SliceDataset(
    root=pathlib.Path(
      'C:/Users/KAgarwal/Dropbox (GaTech)/GATECH OMS ANALYTICS/CS 7643 Deep Learning/Project/fastMRI/singlecoil_train'
    ),
    transform=transforms.singlecoilVarNetDataTransform(mask_func),
    challenge='singlecoil',
    sample_rate=0.01
)


for masked_kspace in dataset:
    # Do reconstruction
    pass