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
