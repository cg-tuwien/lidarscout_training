import importlib
import typing
from collections.abc import Mapping, Sequence

import pytorch_lightning as pl
import torch
from torch import nn


def instantiate_loss_spec(spec) -> typing.Any:
    if spec is None:
        return None

    if isinstance(spec, nn.Module):
        return spec

    if isinstance(spec, Mapping):
        if 'class_path' in spec:
            class_path = spec['class_path']
            module_path, class_name = class_path.rsplit('.', 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            init_args = spec.get('init_args', {}) or {}
            return cls(**{key: instantiate_loss_spec(value) for key, value in init_args.items()})

        return {key: instantiate_loss_spec(value) for key, value in spec.items()}

    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return [instantiate_loss_spec(value) for value in spec]

    return spec


def _broadcast_mask(mask: torch.Tensor, loss_map: torch.Tensor) -> torch.Tensor:
    if mask.ndim == loss_map.ndim:
        return mask
    if mask.ndim == loss_map.ndim + 1:
        return mask[:, 0]
    if mask.ndim == loss_map.ndim - 1:
        return mask.unsqueeze(1) if loss_map.ndim == 4 else mask
    if mask.ndim == 2 and loss_map.ndim == 3:
        return mask
    raise ValueError(f'Cannot broadcast mask with shape {tuple(mask.shape)} to loss map {tuple(loss_map.shape)}')


class LossComponent(pl.LightningModule):
    def __init__(
        self,
        kind: str,
        name: typing.Optional[str] = None,
        target: typing.Literal['hm', 'rgb'] = 'hm',
        weight: float = 1.0,
        use_valid_pixel_mask: bool = False,
        valid_pixel_mask_key: str = 'patch_hm_mask',
        gaussian_resize_to: int = 256,
        lpips_net_type: typing.Literal['alex', 'vgg', 'squeeze'] = 'alex',
        seam_fall_off_factor: float = 5.0,
    ):
        super().__init__()
        self.kind = kind
        self.name = name or kind
        self.target = target
        self.weight = float(weight)
        self.use_valid_pixel_mask = use_valid_pixel_mask
        self.valid_pixel_mask_key = valid_pixel_mask_key
        self.gaussian_resize_to = gaussian_resize_to
        self.lpips_net_type = lpips_net_type
        self.seam_fall_off_factor = seam_fall_off_factor

    def _get_target(self, batch: dict) -> torch.Tensor:
        if self.target == 'hm':
            return batch['hm_gt_ps']
        if self.target == 'rgb':
            return batch['rgb_gt']
        raise ValueError(f'Unsupported target: {self.target}')

    def _get_prediction_slice(self, pred: torch.Tensor) -> torch.Tensor:
        if self.target == 'hm':
            return pred[:, 0:1]
        if self.target == 'rgb':
            if pred.shape[1] < 4:
                raise ValueError('RGB losses require a 4-channel prediction tensor')
            return pred[:, 1:4]
        raise ValueError(f'Unsupported target: {self.target}')

    def _get_valid_mask(self, batch: dict, target: torch.Tensor) -> torch.Tensor:
        if self.use_valid_pixel_mask and self.valid_pixel_mask_key in batch:
            mask = batch[self.valid_pixel_mask_key]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            return mask.to(dtype=target.dtype, device=target.device) > 0.5

        mask = ~torch.isnan(target)
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        return mask

    def _reduce_to_spatial_mean(self, loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _broadcast_mask(mask, loss_map).float()
        valid_count = mask.sum().clamp_min(1e-8)
        return (loss_map * mask).sum() / valid_count

    def _rgb_safe_target(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        target_rgb = target_rgb.clone()
        target_rgb[torch.isnan(target_rgb)] = pred_rgb[torch.isnan(target_rgb)]
        return target_rgb

    def _hm_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        target = self._get_target(batch)
        pred_hm = torch.nan_to_num(self._get_prediction_slice(pred)[:, 0], nan=0.0)
        target_safe = torch.nan_to_num(target, nan=0.0)
        return nn.functional.mse_loss(pred_hm, target_safe, reduction='none')

    def _hm_gradient_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        from source.base.metrics import gradient_loss_masked

        target = self._get_target(batch)
        pred_hm = torch.nan_to_num(self._get_prediction_slice(pred), nan=0.0)
        return gradient_loss_masked(pred_hm, target.unsqueeze(1))[:, 0]

    def _hm_seam_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        loss_hm = self._hm_loss_map(pred, batch)
        res = loss_hm.shape[1]
        pixel_coords_x = torch.arange(res, device=loss_hm.device)
        pixel_coords_y = torch.arange(res, device=loss_hm.device)
        pixel_coords_x, pixel_coords_y = torch.meshgrid(pixel_coords_x, pixel_coords_y, indexing='xy')
        center = (res - 1) / 2
        distances = torch.abs(pixel_coords_x - center) + torch.abs(pixel_coords_y - center)
        dist_norm = distances / res
        dist_norm = torch.maximum(dist_norm * self.seam_fall_off_factor, torch.zeros_like(dist_norm))
        sum_to_one_factor = dist_norm.numel() / torch.sum(dist_norm)
        dist_norm = dist_norm * sum_to_one_factor
        return loss_hm * dist_norm[None].expand_as(loss_hm)

    def _rgb_seam_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        loss_rgb = self._rgb_loss_map(pred, batch, 'mse')
        res = loss_rgb.shape[1]
        pixel_coords_x = torch.arange(res, device=loss_rgb.device)
        pixel_coords_y = torch.arange(res, device=loss_rgb.device)
        pixel_coords_x, pixel_coords_y = torch.meshgrid(pixel_coords_x, pixel_coords_y, indexing='xy')
        center = (res - 1) / 2
        distances = torch.abs(pixel_coords_x - center) + torch.abs(pixel_coords_y - center)
        dist_norm = distances / res
        dist_norm = torch.maximum(dist_norm * self.seam_fall_off_factor, torch.zeros_like(dist_norm))
        sum_to_one_factor = dist_norm.numel() / torch.sum(dist_norm)
        dist_norm = dist_norm * sum_to_one_factor
        return loss_rgb * dist_norm[None].expand_as(loss_rgb)

    def _hm_mean_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        target = self._get_target(batch)
        pred_hm = torch.nan_to_num(self._get_prediction_slice(pred)[:, 0], nan=0.0)
        pred_mean = pred_hm.mean(dim=(1, 2))
        if 'hm_mean' in batch:
            target_mean = batch['hm_mean']
            if target_mean.ndim > 1:
                target_mean = target_mean[:, 0]
        else:
            target_mean = torch.nanmean(target, dim=(1, 2))

        mean_loss = nn.functional.mse_loss(pred_mean, target_mean, reduction='none')
        mean_loss = torch.clip(mean_loss, min=0.0, max=0.01)
        return mean_loss[:, None, None].broadcast_to(target.shape if target.ndim == 3 else target[:, 0].shape)

    def _hm_fft_loss_map(self, pred: torch.Tensor, batch: dict) -> torch.Tensor:
        from source.base.metrics import fft_amplitude_loss

        target = self._get_target(batch)
        pred_hm = torch.nan_to_num(self._get_prediction_slice(pred), nan=0.0)
        return fft_amplitude_loss(pred_hm, target.unsqueeze(1))[:, 0]

    def _rgb_loss_map(self, pred: torch.Tensor, batch: dict, mode: str) -> torch.Tensor:
        pred_rgb = torch.nan_to_num(self._get_prediction_slice(pred), nan=0.0)
        target_rgb = self._get_target(batch)
        target_rgb = self._rgb_safe_target(pred_rgb, target_rgb)

        if mode == 'mse':
            loss_rgb = nn.functional.mse_loss(pred_rgb, target_rgb, reduction='none')
        elif mode == 'huber':
            loss_rgb = nn.functional.huber_loss(pred_rgb, target_rgb, reduction='none')
        elif mode == 'l1':
            loss_rgb = nn.functional.l1_loss(pred_rgb, target_rgb, reduction='none')
        elif mode == 'gradient':
            from source.base.metrics import gradient_loss_masked

            loss_rgb = gradient_loss_masked(pred_rgb, target_rgb)
        elif mode == 'fft':
            from source.base.metrics import fft_amplitude_loss

            loss_rgb = fft_amplitude_loss(pred_rgb, target_rgb)
        elif mode == 'lpips':
            from source.base.metrics import lpips

            loss_rgb = lpips(pred_rgb, target_rgb, net_type=self.lpips_net_type)
            h, w = target_rgb.shape[2], target_rgb.shape[3]
            loss_rgb = loss_rgb[:, None, None].broadcast_to((loss_rgb.shape[0], h, w))
            return loss_rgb
        elif mode == 'ssim':
            from source.base.metrics import ssim

            loss_rgb = 1.0 - ssim(pred_rgb, target_rgb, data_range=1.0)
            h, w = target_rgb.shape[2], target_rgb.shape[3]
            loss_rgb = loss_rgb[:, None, None].broadcast_to((loss_rgb.shape[0], h, w))
            return loss_rgb
        elif mode == 'sparse':
            if 'patch_rgb_rasterize' not in batch:
                raise KeyError('patch_rgb_rasterize is required for sparse RGB loss')

            def _blur_rgb_pts(rgb_pts: torch.Tensor):
                kernel = torch.ones(size=(3, 3, 3, 3), device=rgb_pts.device)
                rgb_pts_shape = rgb_pts.shape
                rgb_pts_shape_flat = (rgb_pts_shape[0] * rgb_pts_shape[1],) + rgb_pts_shape[2:]
                rgb_pts_flat = rgb_pts.view(rgb_pts_shape_flat)
                rgb_pts_blurred = nn.functional.conv2d(rgb_pts_flat, kernel, padding=1)
                return rgb_pts_blurred.view(rgb_pts_shape)

            def weighted_sum_filter(rgb_pts: torch.Tensor):
                rgb_pts = rgb_pts.clone()
                rgb_pts[torch.isnan(rgb_pts)] = 0.0
                rgb_sum = _blur_rgb_pts(rgb_pts)

                rgb_weights = rgb_pts.clone()
                rgb_weights[rgb_weights != 0.0] = 1.0
                rgb_weights = _blur_rgb_pts(rgb_weights)
                return rgb_sum / rgb_weights

            rgb_target_sparse = weighted_sum_filter(batch['patch_rgb_rasterize'].clone().detach())
            diff = (rgb_target_sparse.shape[-1] - target_rgb.shape[-1]) // 2
            rgb_target_sparse = rgb_target_sparse[..., diff:diff + target_rgb.shape[-1], diff:diff + target_rgb.shape[-1]]
            unknown_mask = torch.isnan(rgb_target_sparse)
            rgb_target_sparse[unknown_mask] = 0.0
            loss_rgb = nn.functional.mse_loss(pred_rgb, rgb_target_sparse, reduction='none')
            loss_rgb[unknown_mask] = 0.0
            loss_rgb = loss_rgb.sum(1)

            num_valid = torch.sum(~unknown_mask)
            if num_valid == 0:
                return loss_rgb
            num_possible = rgb_target_sparse.numel()
            scaling_factor = num_possible / num_valid
            loss_rgb = loss_rgb * scaling_factor
            return torch.clip(loss_rgb, min=0.0, max=1.0)
        else:
            raise ValueError(f'Unsupported RGB loss mode: {mode}')

        loss_rgb = torch.clip(loss_rgb, min=0.0, max=1.0)
        return loss_rgb.sum(1)

    def compute_loss_map(self, pred: torch.Tensor, batch: dict, model=None) -> torch.Tensor:
        if self.kind == 'mse':
            return self._hm_loss_map(pred, batch) if self.target == 'hm' else self._rgb_loss_map(pred, batch, 'mse')
        if self.kind == 'gradient':
            return self._hm_gradient_loss_map(pred, batch) if self.target == 'hm' else self._rgb_loss_map(pred, batch, 'gradient')
        if self.kind == 'seam':
            return self._hm_seam_loss_map(pred, batch) if self.target == 'hm' else self._rgb_seam_loss_map(pred, batch)
        if self.kind == 'mean':
            return self._hm_mean_loss_map(pred, batch)
        if self.kind == 'fft':
            return self._hm_fft_loss_map(pred, batch) if self.target == 'hm' else self._rgb_loss_map(pred, batch, 'fft')
        if self.kind == 'huber':
            return self._rgb_loss_map(pred, batch, 'huber')
        if self.kind == 'l1':
            return self._rgb_loss_map(pred, batch, 'l1')
        if self.kind == 'lpips':
            return self._rgb_loss_map(pred, batch, 'lpips')
        if self.kind == 'ssim':
            return self._rgb_loss_map(pred, batch, 'ssim')
        if self.kind == 'sparse':
            return self._rgb_loss_map(pred, batch, 'sparse')
        raise ValueError(f'Unsupported loss kind: {self.kind}')

    def forward(self, pred: torch.Tensor, batch: dict, model=None) -> torch.Tensor:
        return self.compute_loss_map(pred, batch, model=model)


class LossMixer(pl.LightningModule):
    def __init__(
        self,
        losses: typing.Sequence[typing.Any],
        weights: typing.Optional[typing.Sequence[float]] = None,
        learned_weights: bool = False,
        use_valid_pixel_mask: bool = False,
        valid_pixel_mask_key: str = 'patch_hm_mask',
    ):
        super().__init__()
        self.losses = nn.ModuleList([instantiate_loss_spec(loss) for loss in losses])
        self.component_names = [getattr(loss, 'name', f'loss_{index}') for index, loss in enumerate(self.losses)]
        self.use_valid_pixel_mask = use_valid_pixel_mask
        self.valid_pixel_mask_key = valid_pixel_mask_key
        self.learned_weights = learned_weights

        if weights is None:
            weights = [1.0] * len(self.losses)
        if len(weights) != len(self.losses):
            raise ValueError('weights must match the number of losses')

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if learned_weights:
            self.weight_logits = nn.Parameter(weights_tensor.log())
        else:
            self.register_buffer('constant_weights', weights_tensor)

    def _get_weights(self) -> torch.Tensor:
        if self.learned_weights:
            return torch.softmax(self.weight_logits, dim=0)
        return self.constant_weights

    def forward(self, pred: torch.Tensor, batch: dict, model=None):
        loss_maps = []
        loss_means = []

        for loss_module in self.losses:
            loss_map = loss_module(pred, batch, model=model)
            target = loss_module._get_target(batch)
            if self.use_valid_pixel_mask and self.valid_pixel_mask_key in batch:
                valid_mask = batch[self.valid_pixel_mask_key]
                if valid_mask.ndim == 3:
                    valid_mask = valid_mask.unsqueeze(1)
                valid_mask = valid_mask.to(dtype=loss_map.dtype, device=loss_map.device) > 0.5
            else:
                valid_mask = loss_module._get_valid_mask(batch, target)

            if loss_map.ndim == 3 and valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask[:, 0]

            loss_mean = loss_module._reduce_to_spatial_mean(loss_map, valid_mask)
            loss_maps.append(loss_map)
            loss_means.append(loss_mean)

        loss_means_tensor = torch.stack(loss_means)
        weights = self._get_weights().to(loss_means_tensor.device)
        total_loss = torch.sum(loss_means_tensor * weights)
        loss_maps_tensor = torch.stack(loss_maps)
        return total_loss, loss_means_tensor, loss_maps_tensor
