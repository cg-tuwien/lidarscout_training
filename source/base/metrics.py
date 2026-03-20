from operator import gt
import os
import typing

import numpy as np
if typing.TYPE_CHECKING:
    import torch


def rmse(predictions: np.ndarray, targets: np.ndarray):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(prediction: 'torch.Tensor', target: 'torch.Tensor', max_val: float = 255) -> 'torch.Tensor':
    import torch
    mse = torch.mean((prediction - target) ** 2)
    return 10 * torch.log10(max_val ** 2 / mse)

def gradient_rmse(prediction: 'torch.Tensor', target: 'torch.Tensor'):
    import kornia
    import torch
    import torch.nn.functional as F
    # kornia.filters.sobel computes the spatial gradient magnitude
    
    pred_edges = kornia.filters.sobel(prediction)  # [B, C, H, W]
    gt_edges = kornia.filters.sobel(target)
    
    error = (pred_edges - gt_edges)**2
    
    valid_mask = ~torch.isnan(target)
    if valid_mask.sum() > 0:
        # erode mask to get the Sobel filter convolutions to be valid
        kernel = torch.ones((1, 1, 3, 3), device=target.device, dtype=torch.float32)
        mask_float = valid_mask[:, 0:1, :, :].float()
        with torch.no_grad():
            neighborhood_sum = F.conv2d(mask_float, kernel, padding=1)
            eroded_mask = (neighborhood_sum == 9.0)
            
            # Expand back to match channel dimension (e.g., 3 for RGB)
            eroded_mask = eroded_mask.expand_as(target)
        error_valid = error[eroded_mask]
    else:
        error_valid = error.flatten()  # all nans, just return something to avoid error
        
    return torch.sqrt(error_valid.mean())

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
