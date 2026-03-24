from operator import gt
import os
import typing

import numpy as np
if typing.TYPE_CHECKING:
    import torch


def rmse(predictions: np.ndarray, targets: np.ndarray):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(prediction: 'torch.Tensor', target: 'torch.Tensor', max_val: float = 1.0) -> 'torch.Tensor':
    # max_val 1.0 for floats [0, 1], 255.0 for uint8
    import torch
    mse = torch.mean((prediction - target) ** 2)
    psnr_value = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_value

def gradient_rmse(prediction: 'torch.Tensor', target: 'torch.Tensor'):
    """Compute RMSE of Sobel gradients between prediction and target, ignoring NaNs in target.
    :param prediction: tensor of shape (N, C, H, W)
    :param target: tensor of shape (N, C, H, W) with possible NaNs indicating unknown values to ignore
    :return: tensor of shape (N,) with RMSE values for each sample in the batch
    """
    import torch
    
    error = gradient_loss_masked(prediction, target)
    error = error ** 2
    error = torch.sqrt(error.mean())
        
    return error

def gradient_loss_masked(prediction: 'torch.Tensor', target: 'torch.Tensor'):
    """Compute squared error of Sobel gradients between prediction and target, ignoring NaNs in target.
    :param prediction: tensor of shape (N, C, H, W)
    :param target: tensor of shape (N, C, H, W) with possible NaNs indicating unknown values to ignore
    :return: tensor of shape (N, C, H, W) with squared error values for each sample in the batch
    """
    import kornia
    import torch
    import torch.nn.functional as F
    
    pred_edges = kornia.filters.sobel(prediction) 
    gt_edges = kornia.filters.sobel(target)
    
    error = (pred_edges - gt_edges)
    
    nan_mask = torch.isnan(target)
    
    if nan_mask.any():
        # erode mask to get the Sobel filter convolutions to be valid
        kernel = torch.ones((1, 1, 3, 3), device=target.device, dtype=torch.float32)
        mask_float = nan_mask[:, 0:1, :, :].float()
        with torch.no_grad():
            neighborhood_sum = F.conv2d(mask_float, kernel, padding=1)
            eroded_mask = (neighborhood_sum > 0.0)
            
            # Expand back to match channel dimension (e.g., 3 for RGB)
            eroded_mask = eroded_mask.expand_as(target)
        error[eroded_mask] = 0.0
        
        # # Handle case where eroded mask is still empty (NaNs at borders)
        # if error_valid.numel() == 0:
        #     return torch.tensor(0.0, device=target.device, dtype=target.dtype)
        if torch.isnan(error).any():
            raise ValueError('NaNs still present in error after masking')
        
    return error
    

def lpips(prediction: 'torch.Tensor', target: 'torch.Tensor',
          net_type: typing.Literal["vgg", "alex", "squeeze"] = "alex") \
        -> 'torch.Tensor':
    """
    LPIPS metric
    :param prediction: tensor of shape (N, 3, H, W), values in range [0, 1]
    :param target: tensor of shape (N, 3, H, W), values in range [0, 1]
    :param net_type: str indicating backbone network type to use. Choose between 'alex', 'vgg' or 'squeeze'
    :return: tensor of shape (N, 1)
    """
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips_torch
    from torchvision.transforms.functional import resize

    # like c++ static variable in a function
    model_member_str = 'model' + net_type
    if model_member_str not in lpips.__dict__:
        lpips.__dict__[model_member_str] = lpips_torch(net_type=net_type, reduction='sum', normalize=True)

    # move model to same device as input
    if lpips.__dict__[model_member_str].device != prediction.device:
        lpips.__dict__[model_member_str] = lpips.__dict__[model_member_str].to(prediction.device)

    # clip to [0, 1]
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)

    # rescale to 256, AlexNet produces stripes, VGG fails completely
    prediction = resize(prediction, size=[256, 256], antialias=True)
    target = resize(target, size=[256, 256], antialias=True)

    # resolve when issue is fixed: https://github.com/Lightning-AI/torchmetrics/issues/3052
    # lpips_result = lpips.__dict__[model_member_str](prediction, target)  # can't avoid reduction here
    from torchmetrics.functional.image.lpips import _lpips_update
    loss, total = _lpips_update(prediction, target, lpips.__dict__[model_member_str].net, normalize=True)
    return loss

def learned_loss_weighting(loss: 'torch.Tensor', weight: 'torch.Tensor') -> 'torch.Tensor':
    """Learnable multi-task loss weighting from https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    both tensor are of shape [], scalars
    """
    import torch
    weighted_loss = torch.exp(-weight) * loss + weight
    return weighted_loss

def density_weighted_loss(loss_per_pixel: 'torch.Tensor', lin_input: 'torch.Tensor', nn_input: 'torch.Tensor', alpha=5.0):
    """
    Dynamically scales the MSE loss based on the underlying point cloud density.
    Instead of fighting the CNN's urge to smooth, we control where it is allowed to smooth. We use the absolute difference between your Linear and Nearest Neighbor inputs as a real-time density map.
    Where lin and nn are identical (Dense Areas), we multiply the L2 loss by a huge number. The network is forced to perfectly memorize the sharp Voronoi edges of the nn input.
    Where lin and nn are vastly different (Sparse/Interpolated Areas), we multiply the L2 loss by near-zero. The network is given a "free pass" to smooth the terrain to connect the dots safely.
    alpha: Controls how aggressively to ignore sparse areas. Higher = focus only on dense.
    """
    
    import torch
    
    delta = torch.abs(lin_input - nn_input)
    
    if delta.ndim > 3:  # assume channel dim at 1: (B, C, H, W) -> (B, H, W)
            delta = delta.mean(dim=1)
    
    weight_mask = torch.exp(-alpha * delta)
    
    # Normalize the mask so the overall learning rate doesn't collapse
    weight_mask = weight_mask / (torch.mean(weight_mask) + 1e-8)
        
    # Apply density weights and the valid mask
    weighted_loss = loss_per_pixel * weight_mask
    
    return weighted_loss
