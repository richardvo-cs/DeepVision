"""
Evaluation metrics for image deblurring.
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        max_val: Maximum possible pixel value
        
    Returns:
        PSNR value in dB
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure correct shape [H, W, C]
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    if img1.shape[0] == 3:  # [C, H, W] -> [H, W, C]
        img1 = img1.transpose(1, 2, 0)
    if img2.shape[0] == 3:
        img2 = img2.transpose(1, 2, 0)
    
    return peak_signal_noise_ratio(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        
    Returns:
        SSIM value
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure correct shape [H, W, C]
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    if img1.shape[0] == 3:  # [C, H, W] -> [H, W, C]
        img1 = img1.transpose(1, 2, 0)
    if img2.shape[0] == 3:
        img2 = img2.transpose(1, 2, 0)
    
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

