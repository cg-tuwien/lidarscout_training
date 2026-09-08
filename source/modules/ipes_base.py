import os
import abc
import typing

import numpy as np
import torch
from torch import nn

from source.base import fs
from source.base.nn import BaseModule
from source.base.visualization import save_hm_as_pts, save_img_batch, get_vis_params
from source.modules.losses import instantiate_loss_spec


class IpesBase(BaseModule):

    def __init__(self,
                 predict_batch_size, debug, show_unused_params, name,
                 loss_module: typing.Any = None, use_valid_pixel_mask: bool = False,
                 valid_pixel_mask_key: str = 'patch_hm_mask',
                 train_metrics_every_n_steps: int = 1):
        super().__init__(debug, show_unused_params, name)

        # self.lr = 0.001  # for lr tuner, not sure if this is used afterward
        self.test_step_outputs = []
        # self.keys_to_log = frozenset({'hm_rmse_ms',})
        self.keys_to_log = frozenset({'hm_rmse_ms', 'hm_gradient_rmse'})  # 'hm_lpips'
        self.regressor = self.make_regressor()

        self.predict_batch_size: int = predict_batch_size
        # cross-predict_step accumulator for reconstruction chunks of the current point cloud --
        # see predict_step's docstring
        self._predict_accum: typing.Optional[dict] = None
        self.use_valid_pixel_mask = use_valid_pixel_mask
        self.valid_pixel_mask_key = valid_pixel_mask_key
        self.train_metrics_every_n_steps = max(1, int(train_metrics_every_n_steps))
        self.loss_module: typing.Any = instantiate_loss_spec(loss_module)
        if self.loss_module is None:
            raise RuntimeError(
                'No loss module configured. Set model.init_args.loss_module in YAML using '
                'source.modules.losses.LossComponent or source.modules.losses.LossMixer.'
            )
        
        # self.hm_loss_weight = nn.Parameter(torch.zeros(1))
        # self.hm_fft_loss_weight = nn.Parameter(torch.zeros(1))
        # self.hm_grad_loss_weight = nn.Parameter(torch.zeros(1))

    @abc.abstractmethod
    def make_regressor(self) -> typing.Any:
        pass

    @staticmethod
    def compute_loss_hm(pred, batch_data):
        height_target = batch_data['hm_gt_ps']
        unknown_mask = torch.isnan(height_target)
        height_target_safe = torch.nan_to_num(height_target, nan=0.0)
        height_loss = nn.functional.mse_loss(input=pred, target=height_target_safe, reduction='none')
        height_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        height_loss = torch.clip(height_loss, min=0.0, max=1.0)
        return height_loss
    
    @staticmethod
    def compute_loss_gradient(pred, batch_data):
        height_target = batch_data['hm_gt_ps']
        unknown_mask = torch.isnan(height_target)

        gradient_target = torch.where(unknown_mask, pred, height_target)
        gradient_target = torch.gradient(gradient_target, dim=(2, 3))
        gradient_target = torch.sum(torch.stack(gradient_target), dim=0)
        gradient_pred = torch.gradient(pred, dim=(2, 3))
        gradient_pred = torch.sum(torch.stack(gradient_pred), dim=0)
        gradient_loss = nn.functional.mse_loss(input=gradient_pred, target=gradient_target, reduction='none')
        gradient_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        return gradient_loss

    @staticmethod
    def compute_loss_hm_seam(pred: torch.Tensor, batch_data: dict, fall_off_factor=5.0):

        loss_hm = IpesBase.compute_loss_hm(pred, batch_data)
            
        # higher loss weights near border
        res = loss_hm.shape[2]  # assume square
        pixel_coords_x = torch.arange(res, device=loss_hm.device)
        pixel_coords_y = torch.arange(res, device=loss_hm.device)
        pixel_coords_x, pixel_coords_y = torch.meshgrid(pixel_coords_x, pixel_coords_y, indexing='xy')
        center = (res - 1) / 2  # consider zero-based indexing
        distances = torch.abs(pixel_coords_x - center) + torch.abs(pixel_coords_y - center)  # L1 norm
        dist_norm = distances / res  # normalize to 0..1
        dist_norm = torch.maximum(dist_norm * fall_off_factor, torch.zeros_like(dist_norm))
        sum_to_one_factor = dist_norm.numel() / torch.sum(dist_norm)
        dist_norm = dist_norm * sum_to_one_factor  # normalize so that the sum of weights is 1 per pixel

        # repeat for batch dimension
        dist_norm_bc = dist_norm[None].expand_as(loss_hm)  # [b, res, res]

        hm_seam_loss = loss_hm * dist_norm_bc
        return hm_seam_loss

    @staticmethod
    def compute_loss_hm_gradient(pred, batch_data):
        from source.base.metrics import gradient_loss_masked
        hm_target = batch_data['hm_gt_ps'].clone()
        hm_gradient_loss = gradient_loss_masked(pred.unsqueeze(1), hm_target.unsqueeze(1))[:, 0]  # temp channel dim
        hm_gradient_loss = torch.clip(hm_gradient_loss, min=0.0, max=1.0)
        return hm_gradient_loss
    
    @staticmethod
    def compute_loss_mean(pred, batch_data):
        mean_loss = nn.functional.mse_loss(input=pred, target=batch_data['hm_mean'], reduction='none')
        mean_loss = torch.clip(mean_loss, min=0.0, max=0.01)
        mean_loss = torch.broadcast_to(mean_loss[:, :, np.newaxis, np.newaxis], batch_data['hm_gt_ps'].shape)
        return mean_loss
    
    @staticmethod
    def compute_loss_hm_fft(pred, batch_data):
        from source.base.metrics import fft_amplitude_loss
        height_loss = fft_amplitude_loss(pred, batch_data['hm_gt_ps'])
        return height_loss

    @staticmethod
    def slice_center(img: torch.Tensor, res_out: int) -> torch.Tensor:
        res_in = img.shape[1]  # assume square and channels first
        diff = (res_in - res_out) // 2
        img  = img[:, diff:diff + res_out, diff:diff + res_out]
        return img

    def compute_loss(self, pred, batch_data):
        loss_result = self.loss_module(pred, batch_data, model=self)
        if isinstance(loss_result, tuple):
            if len(loss_result) == 3:
                return loss_result
            raise ValueError('Configured loss module returned an unexpected tuple shape')

        loss_map = loss_result
        target = batch_data['hm_gt_ps']
        if self.use_valid_pixel_mask and self.valid_pixel_mask_key in batch_data:
            valid_mask = batch_data[self.valid_pixel_mask_key]
            if valid_mask.ndim == 4:
                valid_mask = valid_mask[:, 0]
            valid_mask = valid_mask > 0.5
        else:
            valid_mask = ~torch.isnan(target[:, 0]) if target.ndim == 4 else ~torch.isnan(target)

        if loss_map.ndim == 4 and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)

        valid_mask = valid_mask.float()
        valid_count = valid_mask.sum().clamp_min(1e-8)
        loss_mean = (loss_map * valid_mask).sum() / valid_count
        return loss_mean, torch.stack([loss_mean]), torch.stack([loss_map])

    def calc_metrics(self, pred, batch):
        pred = pred.detach()
        pred_hm_ps = pred[:, 0]
        pred_hm_ps_flat = pred[:, 0].flatten()
        pred_hm_ms = self.post_proc_pred(batch, pred)
        pred_hm_ms_flat = pred_hm_ms[:, 0].detach().flatten()

        hm_target_ps = batch['hm_gt_ps'].detach()
        hm_target_ps_flat = hm_target_ps.flatten()
        hm_target_ms_flat = batch['hm_gt_ms'].detach().flatten()

        hm_target_ps_nan = torch.isnan(hm_target_ps_flat)
        hm_pred_ps_nan = torch.isnan(pred_hm_ps_flat)
        hm_nan = torch.logical_or(hm_target_ps_nan, hm_pred_ps_nan)

        pred_hm_ps_no_nan = pred_hm_ps_flat[~hm_nan]
        pred_hm_ms_no_nan = pred_hm_ms_flat[~hm_nan]
        height_target_ps_no_nan = hm_target_ps_flat[~hm_nan]
        height_target_ms_no_nan = hm_target_ms_flat[~hm_nan]

        hm_e_ps = pred_hm_ps_no_nan - height_target_ps_no_nan
        hm_e_ms = pred_hm_ms_no_nan - height_target_ms_no_nan

        hm_rmse_ps = torch.sqrt(torch.mean(torch.square(hm_e_ps)))
        hm_rmse_ms = torch.sqrt(torch.mean(torch.square(hm_e_ms)))

        from source.base.metrics import gradient_rmse, lpips
        hm_gradient_rmse = gradient_rmse(pred_hm_ps.unsqueeze(1), hm_target_ps.unsqueeze(1))

        # hm_target_ps_no_nan = hm_target_ps.clone()
        # hm_target_ps_no_nan[hm_nan.view_as(hm_target_ps)] = pred_hm_ps[hm_nan.view_as(pred_hm_ps)]
        # hm_lpips = lpips(pred_hm_ps.unsqueeze(1), hm_target_ps_no_nan.unsqueeze(1), net_type='alex').mean()
            
        eval_dict = {
            'hm_rmse_ms': hm_rmse_ms,
            'hm_rmse_ps': hm_rmse_ps,  # need this for the scheduler
            'hm_gradient_rmse': hm_gradient_rmse,
            # 'hm_lpips': hm_lpips,
        }
        return eval_dict

    def post_proc_pred(self, batch: dict, pred):
        # convert hm from patch space to model space
        from source.base.normalization import hm_patch_space_to_model_space_tensor

        pts_query_ms = batch['pts_query_ms']
        patch_radius_hm_ms = batch['patch_radius_hm_ms'][0].item()
        numerical_stability_factor = batch['numerical_stability_factor'][0].item()

        # do only for hm, not for RGB
        pred_post_proc = pred.clone()
        pred_post_proc = hm_patch_space_to_model_space_tensor(
            hm_to_convert_ps=pred_post_proc, pts_patch_center_ms=pts_query_ms,
            patch_radius_ms=patch_radius_hm_ms, numerical_stability_z_factor=numerical_stability_factor)

        return pred_post_proc

    # this is only for tracing in exporters: to_torchscript(), to_onnx()
    def forward(self, batch):
        # batch = batch['model_kwargs']  # for dynamo ONNX export
        pred = self.regressor.forward(batch)  # only one dict input
        # pred_proc = self.post_proc_pred(batch, pred)  # will be done on SIMLOD side
        return pred

    def should_compute_train_metrics(self, batch_idx: int) -> bool:
        if self.train_metrics_every_n_steps <= 1:
            return True
        return (batch_idx % self.train_metrics_every_n_steps) == 0

    def common_step(self, batch, step: str, batch_idx: typing.Optional[int] = None,
                    compute_metrics: typing.Optional[bool] = None):
        if compute_metrics is None:
            if step == 'train':
                compute_metrics = self.should_compute_train_metrics(batch_idx=batch_idx or 0)
            else:
                compute_metrics = True

        pred = self.regressor.forward(batch)
        loss, loss_components_mean, loss_components = self.compute_loss(pred=pred, batch_data=batch)
        metrics_dict = self.calc_metrics(pred=pred, batch=batch) if compute_metrics else {}

        if bool(self.debug):
            self.visualize_step_results(batch_data=batch, predictions=pred,
                                        losses=loss_components, metrics=metrics_dict,
                                        iteration=self.trainer.global_step, step=step)
        return loss, loss_components_mean, loss_components, metrics_dict, pred

    def training_step(self, batch, batch_idx):
        loss, loss_components_mean, loss_components, metrics_dict, pred = self.common_step(
            batch=batch, step='train', batch_idx=batch_idx)
        self.do_logging(loss, loss_components_mean, log_type='train',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='hm_rmse_ms',
                        log_metrics=bool(metrics_dict))
        return loss

    def validation_step(self, batch, batch_idx):
        from source.base.profiling import get_duration
        duration, step_data = get_duration(self.common_step, {'batch': batch, 'step': 'val'}, warmup=False)
        self.log('epoch/val/duration_s', duration, on_step=False, on_epoch=True,
                 logger=True, batch_size=batch['pts_query_ms'].shape[0])

        from source.base.profiling import get_process_memory_bytes
        self.log('epoch/val/cpu_mem_gb', get_process_memory_bytes() / 1024 / 1024 / 1024,
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])

        from torch.cuda import memory_allocated
        self.log('epoch/val/gpu_mem_gb', memory_allocated() / 1024 / 1024 / 1024,
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])

        # self.log('epoch/val/weights/hm_loss_weight', self.hm_loss_weight.item(),
        #          on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        # self.log('epoch/val/weights/hm_fft_loss_weight', self.hm_fft_loss_weight.item(),
        #          on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        # self.log('epoch/val/weights/hm_grad_loss_weight', self.hm_grad_loss_weight.item(),
        #          on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        
        loss, loss_components_mean, loss_components, metrics_dict, pred = step_data
        self.do_logging(loss, loss_components_mean, log_type='val',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='hm_rmse_ms')
        
        # Log validation losses/metrics aggregated per-epoch so LR schedulers
        # that monitor validation metrics (e.g. ReduceLROnPlateau) can access them.
        # we can't use the learned weights for this because they are not stable
        # therefore, we use a simple sum of the most important loss components, which is the height RMSE and RGB RMSE
        scheduler_target_loss = metrics_dict['hm_rmse_ps'] + (metrics_dict['rgb_rmse'] if 'rgba_rmse' in metrics_dict else 0.0)
        self.log('epoch/sched_target', scheduler_target_loss, on_step=False, on_epoch=True,
                 logger=True, batch_size=batch['pts_query_ms'].shape[0])
        
        return loss

    def test_step(self, batch, batch_idx):
        pred = self.regressor.forward(batch)

        loss, loss_components_mean, loss_components = self.compute_loss(pred=pred, batch_data=batch)
        metrics_dict = self.calc_metrics(pred=pred, batch=batch)

        self.visualize_step_results(batch_data=batch, predictions=pred,
                                    losses=loss_components, metrics=metrics_dict,
                                    iteration=self.trainer.global_step, step='test')

        loss_components_mean = loss_components_mean.squeeze(0)
        loss_components = loss_components.squeeze(0)
        pc_file_in = batch['pc_file_in'][0]

        results = {'pc_file_in': pc_file_in, 'loss': loss,
                   'loss_components_mean': loss_components_mean,
                #    'loss_components': loss_components, 
                   'metrics_dict': metrics_dict}
        self.test_step_outputs.append(results)

        prog_bar = self.get_prog_bar()
        prog_bar.test_progress_bar.set_postfix_str('pc_file: {}'.format(os.path.basename(pc_file_in)), refresh=True)
        return results

    def on_test_epoch_end(self):

        from source.base.evaluation import make_test_report
        from source.base.container import aggregate_dicts, flatten_dicts
        from source.dataloaders.base_data_module import read_shape_list, get_results_dir

        shape_names = read_shape_list(self.in_file)
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)

        outputs_flat = flatten_dicts(self.test_step_outputs)
        metrics_dicts_stacked = aggregate_dicts(outputs_flat, method='stack')

        output_file = os.path.join(results_dir, 'metrics_{}.xlsx'.format(self.name))
        metrics_keys_to_log = ['hm_rmse_ms', 'hm_gradient_rmse', 'hm_lpips']
        loss_total_mean, metrics = make_test_report(
            shape_names=shape_names, results=metrics_dicts_stacked,
            output_file=output_file, output_names=self.output_names, is_dict=True,
            metrics_keys_to_log=frozenset(metrics_keys_to_log))

        hm_rmse_ms_mean = metrics[metrics_keys_to_log.index('hm_rmse_ms')]
        # self.log('epoch/test/RMSE_ms', hm_rmse_ms_mean, on_step=False, on_epoch=True, logger=True)  # avoid overwrite of train run
        print('\nTest results (mean): Loss={}, RMSE_ms={}'.format(
            loss_total_mean, hm_rmse_ms_mean))

    def fix_heightmaps_for_prediction(self, batch: dict) -> dict:
        # reconstruction query points are just 2D
        # need somewhat decent z values as estimate, take middle of HM
        # fix normalization of heightmaps

        hm_key_for_reference = 'patch_hm_linear' if 'patch_hm_linear' in batch.keys() else None
        if hm_key_for_reference is None:
            hm_keys = [k for k in batch.keys() if k.startswith('patch_hm_') and not k.endswith('_mask')]
            if len(hm_keys) == 0:
                raise ValueError('no interpolation method found')
            hm_key_for_reference = hm_keys[0]

        hm_lin_center = batch[hm_key_for_reference].shape[2] // 2
        patch_hm_lin_center_ps = torch.zeros(
            (batch[hm_key_for_reference].shape[0],),
            device=batch[hm_key_for_reference].device,
            dtype=batch[hm_key_for_reference].dtype,
        )
        found_numeric_center = False
        center_area_size = 1
        while not found_numeric_center:
            patch_hm_lin_center = batch[hm_key_for_reference][:, :,
                                  hm_lin_center - center_area_size:hm_lin_center + center_area_size,
                                  hm_lin_center - center_area_size:hm_lin_center + center_area_size]
            patch_hm_lin_center_ps = torch.nanmean(patch_hm_lin_center, dim=(1, 2, 3))
            if torch.all(torch.isfinite(patch_hm_lin_center_ps)):
                found_numeric_center = True
            elif center_area_size > 16:  # too many iterations, let rest deal with NaNs
                break
            else:
                center_area_size += 1

        numerical_stability_factor = batch['numerical_stability_factor'][0].item()
        patch_radius_interp_ms = batch['patch_radius_interp_ms'][0].item()

        patch_hm_lin_center_ms = patch_hm_lin_center_ps / numerical_stability_factor * patch_radius_interp_ms

        batch['pts_query_ms'][..., 2] = patch_hm_lin_center_ms
        for k in batch.keys():
            if k.startswith('patch_hm_') and not k.endswith('_mask'):
                batch[k] = batch[k] - patch_hm_lin_center_ps[:, None, None, None]

        return batch

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0):
        # Reconstruct one point cloud -- but each call only ever receives ONE small chunk of its
        # reconstruction query grid (see IpesDataset._reconstruction_chunk_names /
        # _make_rec_data), generated lazily by the dataloader rather than the whole grid's local
        # point subsamples being resolved eagerly up front (that eager version crashed at
        # ~19GB+ on a 100x-densified stress-test shape). So results are accumulated across calls
        # in self._predict_accum, keyed by nothing but call order (one point cloud is always
        # processed start-to-finish before the next, since chunks of the same point cloud are
        # contiguous in the dataset) -- reset when a chunk's rec_chunk_start_id is 0 (first chunk
        # of a new point cloud), flushed to disk only once rec_is_last_chunk is True. For a point
        # cloud small enough to fit in a single chunk, first-chunk and last-chunk are the same
        # call and this collapses back to the old immediate reset-process-write behavior.
        from source.dataloaders.base_data_module import get_results_dir
        from source.dataloaders.ipes_data_loader import hm_to_pts

        if len(batch['pc_file_in']) > 1:
            raise NotImplementedError('batch size > 1 not supported')

        batch = self.fix_heightmaps_for_prediction(batch)

        pc_file_in = batch['pc_file_in'][0]
        pts_query_ms = batch['pts_query_ms'].detach().cpu().numpy()
        meters_per_pixel = batch['meters_per_pixel'][0].item()
        num_query_pts = pts_query_ms.shape[0]

        is_first_chunk = int(batch['rec_chunk_start_id'][0].item()) == 0
        is_last_chunk = bool(batch['rec_is_last_chunk'][0].item())

        if is_first_chunk or self._predict_accum is None:
            self._predict_accum = {
                'hm_ms': [], 'hm_pts_ms': [], 'hm_pts_norm': [], 'pts_query_ids_xy': [],
                'interp_rgb': [], 'iteration_offset': 0,
            }
        accum = self._predict_accum

        if num_query_pts == 0:
            # every query point in this chunk was filtered out by _make_local_sub_samples (no
            # local points found within the patch radius) -- happens for a chunk that falls
            # entirely in a corner/edge of the reconstruction grid's rectangular bounding box
            # where the point cloud doesn't actually have coverage. Nothing to forward through
            # the model or append; just carry the accumulator through to the next chunk (or, if
            # this is also the last chunk, flush whatever earlier chunks already contributed).
            if not is_last_chunk:
                return 0
        else:
            self._predict_step_process_chunk(batch, pts_query_ms, num_query_pts, meters_per_pixel, accum, hm_to_pts)

        if not is_last_chunk:
            return 0

        return self._predict_step_flush(pc_file_in, accum, get_results_dir)

    def _predict_step_process_chunk(self, batch, pts_query_ms, num_query_pts, meters_per_pixel, accum, hm_to_pts):
        prog_bar = self.get_prog_bar()
        pts_query_ids_xy = batch['pts_query_ids_xy'].detach().cpu().numpy()

        # cache for this chunk's predicted heightmaps
        hm_ms_chunk = None
        hm_pts_ms_chunk = []
        hm_pts_norm_chunk = []

        num_sections = num_query_pts // self.predict_batch_size
        if num_sections == 0:
            patch_ids_chunked = [np.arange(num_query_pts)]
        else:
            patch_ids_chunked = np.array_split(np.arange(num_query_pts), num_sections)
        for iteration, chunk_ids in enumerate(patch_ids_chunked):
            data_chunk = {
                'pts_query_ms': batch['pts_query_ms'][chunk_ids],
                'patch_radius_hm_ms': batch['patch_radius_hm_ms'],
                'numerical_stability_factor': batch['numerical_stability_factor'],
                'sun_pos_xy': batch['sun_pos_xy'][chunk_ids],
                }
            for k in batch.keys():
                if k.startswith('patch_hm_') or k.startswith('patch_rgb_'):
                    data_chunk[k] = batch[k][chunk_ids]

            pred_chunk_hm_ps = self.regressor.forward(data_chunk)
            pred_chunk_hm_ms = self.post_proc_pred(data_chunk, pred_chunk_hm_ps)
            pred_chunk_hm_ms = pred_chunk_hm_ms.detach().cpu().numpy()

            if hm_ms_chunk is None:  # init buffer here to get correct channel count
                out_channels = pred_chunk_hm_ms.shape[1]
                hm_ms_chunk_shape = (num_query_pts, out_channels, self.hm_size, self.hm_size)
                hm_ms_chunk = np.zeros(hm_ms_chunk_shape, dtype=np.float32)
            hm_ms_chunk[chunk_ids] = pred_chunk_hm_ms

            if bool(self.debug):
                data_chunk['patch_radius_interp_ms'] = batch['patch_radius_interp_ms']
                data_chunk['pts_query_ids_xy'] = batch['pts_query_ids_xy'][:, chunk_ids]
                self.visualize_step_results(batch_data=data_chunk, predictions=pred_chunk_hm_ps,
                                            losses=None, metrics=None,
                                            iteration=accum['iteration_offset'] + iteration, step='predict')

            for p in range(pred_chunk_hm_ms.shape[0]):
                pts_hm_ms, pts_normals = hm_to_pts(
                    pred_chunk_hm_ms[p, 0], pts_query_ms[chunk_ids][p], pixel_size=meters_per_pixel)
                hm_pts_ms_chunk.append(pts_hm_ms)
                hm_pts_norm_chunk.append(pts_normals)
            prog_bar.predict_progress_bar.set_postfix_str(
                'chunk@{}, iter {}'.format(int(batch['rec_chunk_start_id'][0].item()), iteration), refresh=True)

        accum['iteration_offset'] += len(patch_ids_chunked)
        accum['hm_ms'].append(hm_ms_chunk)
        accum['hm_pts_ms'].append(np.concatenate(hm_pts_ms_chunk, axis=0))
        accum['hm_pts_norm'].append(np.concatenate(hm_pts_norm_chunk, axis=0))
        accum['pts_query_ids_xy'].append(pts_query_ids_xy)

        # accumulate interpolation-based RGB too (unflattened, per-window images) -- used as a
        # color fallback below only if the model itself doesn't predict color, decided once all
        # chunks are in, so it must be collected on every chunk regardless
        interp_rgb_keys = [k for k in batch.keys() if k.startswith('patch_rgb')]
        if interp_rgb_keys:
            from source.base.img import slice_img_center
            pts_rgb_chunk = batch[interp_rgb_keys[0]].detach().cpu().numpy()
            if pts_rgb_chunk.shape[-1] != self.hm_size or pts_rgb_chunk.shape[-2] != self.hm_size:
                pts_rgb_chunk = slice_img_center(pts_rgb_chunk, self.hm_interp_size, self.hm_size)
            accum['interp_rgb'].append(pts_rgb_chunk.transpose(0, 2, 3, 1))

    def _predict_step_flush(self, pc_file_in, accum, get_results_dir):
        self._predict_accum = None  # release before writing output, done with this point cloud

        if not accum['hm_ms']:
            # every chunk of this point cloud came back with zero valid query points -- nothing
            # to reconstruct (e.g. an entirely-uncovered reconstruction area). Skip writing files
            # rather than crashing on an empty concatenate.
            print(f'WARNING: no valid reconstruction windows for {pc_file_in}, skipping output.')
            return 0

        hm_ms_all = np.concatenate(accum['hm_ms'], axis=0)
        hm_pts_ms_all = np.concatenate(accum['hm_pts_ms'], axis=0)
        hm_pts_norm_all = np.concatenate(accum['hm_pts_norm'], axis=0)
        pts_query_ids_xy_all = np.concatenate(accum['pts_query_ids_xy'], axis=0)
        interp_rgb_all = accum['interp_rgb']

        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        out_file_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '.npy')
        out_file_rgb_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '_rgb' + '.npy')
        out_file_qids_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '_xy' + '.npy')
        out_file_rec = os.path.join(results_dir, 'meshes', os.path.basename(pc_file_in) + '.ply')

        pred_colors = hm_ms_all.shape[1] >= 4
        if pred_colors:  # add color from prediction if available
            pts_rgb = hm_ms_all[:, 1:4]  # * 255.0
            # pts_rgb = np.clip(pts_rgb, 0.0, 255.0)
            pts_rgb = np.clip(pts_rgb, 0.0, 1.0)
            pts_rgb_flat = pts_rgb.transpose(0, 2, 3, 1).reshape(-1, 3)
            hm_ms_all = hm_ms_all[:, 0:1]
        elif interp_rgb_all:  # add color from interpolation if available
            pts_rgb = np.concatenate(interp_rgb_all, axis=0)
            pts_rgb_flat = pts_rgb.reshape(-1, 3)
        else:
            pts_rgb = None
            pts_rgb_flat = None

        # save heightmaps numpy file and point clouds
        fs.make_dir_for_file(out_file_np)
        np.save(out_file_np, hm_ms_all)
        if pts_rgb is not None:
            np.save(out_file_rgb_np, pts_rgb)
        np.save(out_file_qids_np, pts_query_ids_xy_all)

        fs.make_dir_for_file(out_file_rec)
        from source.base.point_cloud import write_ply
        write_ply(file_path=out_file_rec, points=hm_pts_ms_all, normals=hm_pts_norm_all, colors=pts_rgb_flat)

        return 0  # return something to suppress warning

    def visualize_step_results(self, batch_data: dict, predictions, losses, metrics, iteration: int, step: str):
        if step == 'train' and not self.debug:
            return  # no visualization

        from source.dataloaders.base_data_module import get_results_dir
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        results_dir = os.path.join(results_dir, step)

        # prediction in
        pred_np = predictions.detach().cpu().numpy().astype(np.float32)[:, 0]  # height

        vis_params = get_vis_params(batch_data, step)
        vis_batches_range, hm_finite, norm_min, norm_max = vis_params

        from source.base.visualization import images_to_figure

        # patch chunk points, jagged list of list of arrays
        if 'pts_local_ms' in batch_data:
            from source.base.point_cloud import write_ply
            from source.base.visualization import get_vis_file
            batch_size = len(batch_data['pts_local_ms'][0])
            for b in range(batch_size):
                pts_ms_file = get_vis_file(batch_data, 'pts_ms', step, iteration, results_dir, b) + '.ply'
                pts_ps_file = get_vis_file(batch_data, 'pts_ps', step, iteration, results_dir, b) + '.ply'

                pts_local_ms = batch_data['pts_local_ms'][0][b]
                pts_local_ps = batch_data['pts_local_ps'][0][b]
                pts_local_rgb = batch_data['pts_local_rgb'][0][b]
                pts_local_ps[:, 2] /= batch_data['numerical_stability_factor'][0].item()
                write_ply(file_path=pts_ms_file, points=pts_local_ms, colors=pts_local_rgb)
                write_ply(file_path=pts_ps_file, points=pts_local_ps, colors=pts_local_rgb)

        fig_io_imgs = []

        # input heightmaps
        for k in batch_data.keys():
            if k.startswith('patch_hm_'):
                hm_pts = batch_data[k].detach().cpu().numpy()[:, 0]
                hm_input_imgs = save_img_batch(
                    batch_data=batch_data, arr=hm_pts, name='hm_' + k, step=step, iteration=iteration,
                    results_dir=results_dir, vis_batches_range=vis_batches_range,
                    norm_min=norm_min, norm_max=norm_max)
                fig_io_imgs.append(hm_input_imgs)

        # prediction
        pred_hm_imgs = save_img_batch(
            batch_data=batch_data, arr=pred_np, name='hm_pred', step=step, iteration=iteration,
            results_dir=results_dir, vis_batches_range=vis_batches_range,
            norm_min=norm_min, norm_max=norm_max)
        fig_io_imgs.append(pred_hm_imgs)

        save_hm_as_pts(
            name='hm_pred_pts', step=step, iteration=iteration,
            hm_tensor_ps=predictions[:, 0], batch_data=batch_data, results_dir=results_dir,
            vis_batches_range=vis_batches_range, color_factor=0.5,
            patch_radius=batch_data['patch_radius_hm_ms'][0].item())

        # target and loss
        if step != 'predict':
            save_hm_as_pts(
                name='hm_gt_pts', step=step, iteration=iteration,
                hm_tensor_ps=batch_data['hm_gt_ps'], batch_data=batch_data, results_dir=results_dir,
                vis_batches_range=vis_batches_range, color_factor=0.25,
                patch_radius=batch_data['patch_radius_hm_ms'][0].item())

            hm_gt_imgs = save_img_batch(
                batch_data=batch_data, arr=hm_finite, name='hm_gt', step=step, iteration=iteration,
                results_dir=results_dir, vis_batches_range=vis_batches_range,
                norm_min=norm_min, norm_max=norm_max)
            fig_io_imgs.append(hm_gt_imgs)

            loss_vis_factor = 10.0
            losses_np = losses[0].detach().cpu().numpy().astype(np.float32)
            losses_np = np.clip(losses_np * loss_vis_factor, 0.0, 1.0)
            loss_imgs = save_img_batch(
                batch_data=batch_data, arr=losses_np, name='hm_loss', step=step, iteration=iteration,
                results_dir=results_dir, vis_batches_range=vis_batches_range)

            # make figures
            fig_loss_imgs = list(zip(pred_hm_imgs, hm_gt_imgs, loss_imgs))
            fig_loss = [p.replace('hm_gt', '0_hm_fig_loss') for p in hm_gt_imgs]
            for img_tuple, fig_img in zip(fig_loss_imgs, fig_loss):
                images_to_figure(img_tuple, fig_img)

            fig_inputs_imgs = list(zip(*fig_io_imgs))
            fig_inputs = [p.replace('hm_pred', '0_hm_fig_input') for p in pred_hm_imgs]
            for img_tuple, fig_img in zip(fig_inputs_imgs, fig_inputs):
                images_to_figure(img_tuple, fig_img)
