import time
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sys
import numpy as np
sys.path.append("./groups")

def unnormalize(sample_dicom, x):
    ds = sample_dicom.copy()
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    ymin = -200
    ymax = 1600
    pixel_min = ymin #/ slope + intercept
    pixel_max = ymax #/ slope + intercept

    scaled_array = (x)*6-700
    unnormalized_array = (scaled_array - intercept) / slope 
    return unnormalized_array


def write_dicom(sample_dicom, array, paths, tt):
    ds = sample_dicom.copy()
    ds.WindowWidth = 400
    ds.WindowCenter = 50
    array = np.transpose(array, (2, 1, 0))[:, :, 0]
    array = unnormalize(sample_dicom, array)
    array = np.clip(array, 0, 2 ** 16 - 1)
    ds.PixelData = array.astype(np.uint16).tobytes()
    return ds.save_as(paths + "/" + str(tt) + ".dcm")