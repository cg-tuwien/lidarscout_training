import os
import abc

import numpy as np
import torch
from torch import nn

from source.base import fs
from source.base.nn import BaseModule
from source.base.visualization import save_hm_as_pts, save_img_batch, get_vis_params


class IpesBase(BaseModule):

    def __init__(self,
                 predict_batch_size, debug, show_unused_params, name):
        super().__init__(debug, show_unused_params, name)

        # self.lr = 0.001  # for lr tuner, not sure if this is used afterward
        self.test_step_outputs = []
        self.keys_to_log = frozenset({'abs_dist_rmse_ps', 'abs_dist_rmse_ms'})
        self.regressor = self.make_regressor()

        self.predict_batch_size = predict_batch_size

    @abc.abstractmethod
    def make_regressor(self):
        pass

    @staticmethod
    def compute_loss_hm(pred, batch_data):
        height_target = batch_data['hm_gt_ps'].clone()
        unknown_mask = torch.isnan(height_target)
        height_target[unknown_mask] = 0.0
        height_loss = nn.functional.mse_loss(input=pred, target=height_target, reduction='none')
        height_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        height_loss = torch.clip(height_loss, min=0.0, max=1.0)
        return height_loss

    @staticmethod
    def compute_loss_gradient(pred, batch_data):
        height_target = batch_data['hm_gt_ps'].clone()
        unknown_mask = torch.isnan(height_target)

        gradient_target = batch_data['hm_gt_ps'].clone()
        gradient_target[unknown_mask] = pred[unknown_mask]
        gradient_target = torch.gradient(gradient_target, dim=(2, 3))
        gradient_target = torch.sum(torch.stack(gradient_target), dim=0)
        gradient_pred = torch.gradient(pred, dim=(2, 3))
        gradient_pred = torch.sum(torch.stack(gradient_pred), dim=0)
        gradient_loss = nn.functional.mse_loss(input=gradient_pred, target=gradient_target, reduction='none')
        gradient_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        return gradient_loss

    @staticmethod
    def compute_loss_hm_seam(loss_hm: torch.Tensor):

        # higher loss weights near border
        res = loss_hm.shape[2]  # assume square
        pixel_coords_x = torch.arange(res, device=loss_hm.device)
        pixel_coords_y = torch.arange(res, device=loss_hm.device)
        pixel_coords_x, pixel_coords_y = torch.meshgrid(pixel_coords_x, pixel_coords_y, indexing='xy')
        center = (res - 1) / 2  # consider zero-based indexing
        distances = torch.abs(pixel_coords_x - center) + torch.abs(pixel_coords_y - center)  # L1 norm
        dist_norm = distances / res  # normalize to 0..1
        fall_off_factor = 5.0
        dist_norm = torch.maximum(dist_norm * fall_off_factor, torch.zeros_like(dist_norm))
        sum_to_one_factor = dist_norm.numel() / torch.sum(dist_norm)
        dist_norm = dist_norm * sum_to_one_factor  # normalize so that the sum of weights is 1 per pixel

        # repeat for batch dimension
        dist_norm_bc = dist_norm[None].expand_as(loss_hm)  # [b, res, res]

        hm_seam_loss = loss_hm * dist_norm_bc
        return hm_seam_loss

    @staticmethod
    def compute_loss_mean(pred, batch_data):
        mean_loss = nn.functional.mse_loss(input=pred, target=batch_data['hm_mean'], reduction='none')
        mean_loss = torch.clip(mean_loss, min=0.0, max=0.01)
        mean_loss = torch.broadcast_to(mean_loss[:, :, np.newaxis, np.newaxis], batch_data['hm_gt_ps'].shape)
        return mean_loss

    def compute_loss(self, pred, batch_data):
        import math

        # keep same order as in yaml
        loss_components = [
            IpesBase.compute_loss_hm(pred, batch_data),
            # IpesBase.compute_loss_gradient(pred, batch_data),
            # self.compute_loss_hm_seam(hm_loss),
            # IpesBase.compute_loss_mean(pred, batch_data),
        ]

        loss_components_mean = [torch.mean(loss) for loss in loss_components]

        loss_components = torch.stack(loss_components)
        loss_components_mean = torch.stack(loss_components_mean)
        loss_tensor = loss_components_mean.mean()

        if math.isclose(loss_tensor.item(), 0.0):
            print('loss is close to zero')

        if math.isnan(loss_tensor.item()):
            print('loss is nan')

        return loss_tensor, loss_components_mean, loss_components

    def calc_metrics(self, pred, batch):
        pred = pred.detach()
        pred_hm_ps = pred[:, 0].reshape(-1)
        pred_proc = self.post_proc_pred(batch, pred)
        pred_hm_ms = pred_proc[:, 0].detach().reshape(-1)

        hm_target_ps = batch['hm_gt_ps'].detach().reshape(-1)
        hm_target_ms = batch['hm_gt_ms'].detach().reshape(-1)

        hm_target_ps_nan = torch.isnan(hm_target_ps)
        hm_pred_ps_nan = torch.isnan(pred_hm_ps)
        hm_nan = torch.logical_or(hm_target_ps_nan, hm_pred_ps_nan)

        pred_hm_ps_no_nan = pred_hm_ps[~hm_nan]
        pred_hm_ms_no_nan = pred_hm_ms[~hm_nan]
        height_target_ps_no_nan = hm_target_ps[~hm_nan]
        height_target_ms_no_nan = hm_target_ms[~hm_nan]

        hm_e_ps = pred_hm_ps_no_nan - height_target_ps_no_nan
        hm_e_ms = pred_hm_ms_no_nan - height_target_ms_no_nan

        hm_rmse_ps = torch.sqrt(torch.mean(torch.square(hm_e_ps)))
        hm_rmse_ms = torch.sqrt(torch.mean(torch.square(hm_e_ms)))

        eval_dict = {'abs_dist_rmse_ms': hm_rmse_ms,
                     'abs_dist_rmse_ps': hm_rmse_ps,}
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

    def common_step(self, batch, step: str):
        pred = self.regressor.forward(batch)
        loss, loss_components_mean, loss_components = self.compute_loss(pred=pred, batch_data=batch)
        metrics_dict = self.calc_metrics(pred=pred, batch=batch)

        if bool(self.debug):
            self.visualize_step_results(batch_data=batch, predictions=pred,
                                        losses=loss_components, metrics=metrics_dict,
                                        iteration=self.trainer.global_step, step=step)
        return loss, loss_components_mean, loss_components, metrics_dict, pred

    def training_step(self, batch, batch_idx):
        loss, loss_components_mean, loss_components, metrics_dict, pred = self.common_step(
            batch=batch, step='train')
        self.do_logging(loss, loss_components_mean, log_type='train',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='abs_dist_rmse_ms')
        return loss

    def validation_step(self, batch, batch_idx):
        from source.base.profiling import get_duration
        duration, step_data = get_duration(self.common_step, {'batch': batch, 'step': 'val'})
        self.log('epoch/val/duration_s', duration, on_step=False, on_epoch=True,
                 logger=True, batch_size=batch['pts_query_ms'].shape[0])

        from source.base.profiling import get_process_memory_bytes
        self.log('epoch/val/cpu_mem_gb', get_process_memory_bytes() / 1024 / 1024 / 1024,
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])

        from torch.cuda import memory_allocated
        self.log('epoch/val/gpu_mem_gb', memory_allocated() / 1024 / 1024 / 1024,
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])

        loss, loss_components_mean, loss_components, metrics_dict, pred = step_data
        self.do_logging(loss, loss_components_mean, log_type='val',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='abs_dist_rmse_ms')
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
                   'loss_components': loss_components, 'metrics_dict': metrics_dict}
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
        loss_total_mean, metrics = make_test_report(
            shape_names=shape_names, results=metrics_dicts_stacked,
            output_file=output_file, output_names=self.output_names, is_dict=True,
            metrics_keys_to_log=frozenset(['abs_dist_rmse_ps', 'abs_dist_rmse_ms']))

        abs_dist_rmse_ps_mean = metrics[0]
        abs_dist_rmse_ms_mean = metrics[1]
        self.log('epoch/test/RMSE_ms', abs_dist_rmse_ms_mean, on_step=False, on_epoch=True, logger=True)
        print('\nTest results (mean): Loss={}, RMSE={}, RMSE_ms={}'.format(
            loss_total_mean, abs_dist_rmse_ps_mean, abs_dist_rmse_ms_mean))

    def fix_heightmaps_for_prediction(self, batch: dict) -> dict:
        # reconstruction query points are just 2D
        # need somewhat decent z values as estimate, take middle of HM
        # fix normalization of heightmaps

        hm_key_for_reference = 'patch_hm_linear' if 'patch_hm_linear' in batch.keys() else None
        if hm_key_for_reference is None:
            hm_keys = [k for k in batch.keys() if k.startswith('patch_hm_')]
            if len(hm_keys) == 0:
                raise ValueError('no interpolation method found')
            hm_key_for_reference = hm_keys[0]

        hm_lin_center = batch[hm_key_for_reference].shape[2] // 2
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
            if k.startswith('patch_hm_'):
                batch[k] = batch[k] - patch_hm_lin_center_ps[:, None, None, None]

        return batch

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0):
        # reconstruct one point cloud
        from source.dataloaders.base_data_module import get_results_dir
        from source.dataloaders.ipes_data_loader import hm_to_pts

        prog_bar = self.get_prog_bar()

        if len(batch['pc_file_in']) > 1:
            raise NotImplementedError('batch size > 1 not supported')

        batch = self.fix_heightmaps_for_prediction(batch)

        pc_file_in = batch['pc_file_in'][0]
        pts_query_ms = batch['pts_query_ms'].detach().cpu().numpy()
        pts_query_ids_xy = batch['pts_query_ids_xy'].detach().cpu().numpy()
        meters_per_pixel = batch['meters_per_pixel'][0].item()
        num_query_pts = pts_query_ms.shape[0]

        # cache for predicted heightmaps
        hm_ms_all = None
        hm_pts_ms_all = []
        hm_pts_norm_all = []

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
                }
            for k in batch.keys():
                if k.startswith('patch_hm_') or k.startswith('patch_rgb_'):
                    data_chunk[k] = batch[k][chunk_ids]

            pred_chunk_hm_ps = self.regressor.forward(data_chunk)
            pred_chunk_hm_ms = self.post_proc_pred(data_chunk, pred_chunk_hm_ps)
            pred_chunk_hm_ms = pred_chunk_hm_ms.detach().cpu().numpy()

            if hm_ms_all is None:  # init buffer here to get correct channel count
                out_channels = pred_chunk_hm_ms.shape[1]
                hm_ms_all_shape = (num_query_pts, out_channels, self.hm_size, self.hm_size)
                hm_ms_all = np.zeros(hm_ms_all_shape, dtype=np.float32)
            hm_ms_all[chunk_ids] = pred_chunk_hm_ms

            if bool(self.debug):
                data_chunk['patch_radius_interp_ms'] = batch['patch_radius_interp_ms']
                data_chunk['pts_query_ids_xy'] = batch['pts_query_ids_xy'][:, chunk_ids]
                self.visualize_step_results(batch_data=data_chunk, predictions=pred_chunk_hm_ps,
                                            losses=None, metrics=None,
                                            iteration=iteration, step='predict')

            for p in range(pred_chunk_hm_ms.shape[0]):
                pts_hm_ms, pts_normals = hm_to_pts(
                    pred_chunk_hm_ms[p, 0], pts_query_ms[chunk_ids][p], pixel_size=meters_per_pixel)
                hm_pts_ms_all.append(pts_hm_ms)
                hm_pts_norm_all.append(pts_normals)
            prog_bar.predict_progress_bar.set_postfix_str('iter: {}'.format(iteration), refresh=True)

        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        out_file_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '.npy')
        out_file_rgb_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '_rgb' + '.npy')
        out_file_qids_np = os.path.join(results_dir, 'npy', os.path.basename(pc_file_in) + '_xy' + '.npy')
        out_file_rec = os.path.join(results_dir, 'meshes', os.path.basename(pc_file_in) + '.ply')

        hm_pts_ms_all = np.concatenate(hm_pts_ms_all, axis=0)
        hm_pts_norm_all = np.concatenate(hm_pts_norm_all, axis=0)

        pred_colors = hm_ms_all.shape[1] >= 4
        if pred_colors:  # add color from prediction if available
            pts_rgb = hm_ms_all[:, 1:4]  # * 255.0
            # pts_rgb = np.clip(pts_rgb, 0.0, 255.0)
            pts_rgb = np.clip(pts_rgb, 0.0, 1.0)
            pts_rgb_flat = pts_rgb.transpose(0, 2, 3, 1).reshape(-1, 3)
            hm_ms_all = hm_ms_all[:, 0:1]
        elif any(k.startswith('patch_rgb') for k in batch.keys()):  # add color from interpolation if available
            # preference in this order
            available_rgb_keys = [k for k in batch.keys() if k.startswith('patch_rgb')]
            chosen_rgb_key = available_rgb_keys[0]  # always take the first one
            from source.base.img import slice_img_center
            pts_rgb = batch[chosen_rgb_key].detach().cpu().numpy()
            if pts_rgb.shape[-1] != self.hm_size or pts_rgb.shape[-2] != self.hm_size:
                pts_rgb = slice_img_center(pts_rgb, self.hm_interp_size, self.hm_size)
            pts_rgb = pts_rgb.transpose(0, 2, 3, 1)
            pts_rgb_flat = pts_rgb.reshape(-1, 3)
        else:
            pts_rgb = None
            pts_rgb_flat = None

        # save heightmaps numpy file and point clouds
        fs.make_dir_for_file(out_file_np)
        np.save(out_file_np, hm_ms_all)
        if pts_rgb is not None:
            np.save(out_file_rgb_np, pts_rgb)
        np.save(out_file_qids_np, pts_query_ids_xy)

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
