import os

import numpy as np
import torch
from torch import nn

from source.base.visualization import save_hm_as_pts, save_img_batch, get_vis_params

from source.modules.ipes_base import IpesBase


class IpesRgbd(IpesBase):

    def __init__(self,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name):

        self.has_color_input = has_color_input
        self.has_color_output = has_color_output

        super().__init__(predict_batch_size, debug, show_unused_params, name)
        
        # self.keys_to_log = self.keys_to_log.union(frozenset({'rgb_psnr', }))
        self.keys_to_log = self.keys_to_log.union(frozenset({'rgb_psnr', 'rgb_gradient_rmse'}))
        
        self.rgb_loss_weight = nn.Parameter(torch.zeros(1))
        self.rgb_fft_loss_weight = nn.Parameter(torch.zeros(1))
        # self.rgb_grad_loss_weight = nn.Parameter(torch.zeros(1))
        

    def compute_loss_rgb_sparse(self, pred, batch_data):
        # 'works' with only query points, no GT RGB maps necessary
        # ignore predictions without GT RGB
        # zero loss if there is no RGB

        def weighted_sum_filter(rgb_pts: torch.Tensor):
            rgb_pts[torch.isnan(rgb_pts)] = 0.0

            rgb_sum = _blur_rgb_pts(rgb_pts)

            rgb_weights = rgb_pts.clone()
            rgb_weights[rgb_weights != 0.0] = 1.0
            rgb_weights = _blur_rgb_pts(rgb_weights)
            return rgb_sum / rgb_weights

        def _blur_rgb_pts(rgb_pts: torch.Tensor):
            # blur RGB points
            # rgb_pts: (B, 3, H, W)
            # return: (B, 3, H, W)
            kernel = torch.ones(size=(3, 3, 3, 3), device=rgb_pts.device)
            rgb_pts_shape = rgb_pts.shape
            rgb_pts_shape_flat = (rgb_pts_shape[0] * rgb_pts_shape[1],) + rgb_pts_shape[2:]
            rgb_pts_flat = rgb_pts.view(rgb_pts_shape_flat)
            rgb_pts_blurred = nn.functional.conv2d(rgb_pts_flat, kernel, padding=1)
            return rgb_pts_blurred.view(rgb_pts_shape)

        rgb_target = weighted_sum_filter(batch_data['patch_rgb_rasterize'].clone().detach())
        diff = (self.hm_interp_size - self.hm_size) // 2  # slice center
        rgb_target = rgb_target[..., diff:diff + self.hm_size, diff:diff + self.hm_size]

        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = 0.0
        rgb_loss = nn.functional.mse_loss(input=pred, target=rgb_target, reduction='none')
        rgb_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        rgb_loss = rgb_loss.sum(2)  # sum over RGB channels

        # scale mean to 1.0
        num_valid = torch.sum(~unknown_mask)
        if num_valid == 0:  # avoid div by zero
            return rgb_loss
        num_possible = rgb_target.numel()
        scaling_factor = num_possible / num_valid
        rgb_loss = rgb_loss * scaling_factor
        rgb_loss = torch.clip(rgb_loss, min=0.0, max=1.0)
        return rgb_loss

    @staticmethod
    def compute_loss_rgb(pred, batch_data):
        rgb_target = batch_data['rgb_gt'].clone()
        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = 0.0
        rgb_loss = nn.functional.mse_loss(input=pred, target=rgb_target, reduction='none')
        rgb_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        rgb_loss = torch.clip(rgb_loss, min=0.0, max=1.0)
        rgb_loss = rgb_loss.sum(1)  # sum over RGB channels
        return rgb_loss

    @staticmethod
    def compute_loss_rgb_huber(pred, batch_data):
        rgb_target = batch_data['rgb_gt'].clone()
        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = 0.0
        rgb_loss = nn.functional.huber_loss(input=pred, target=rgb_target, reduction='none')
        rgb_loss[unknown_mask] = 0.0  # ignore nan (unknown GT)
        rgb_loss = torch.clip(rgb_loss, min=0.0, max=1.0)
        rgb_loss = rgb_loss.sum(1)  # sum over RGB channels
        return rgb_loss

    @staticmethod
    def compute_loss_rgb_l1(pred, batch_data):
        rgb_target = batch_data['rgb_gt'].clone()
        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = pred[unknown_mask]
        rgb_loss = nn.functional.l1_loss(input=pred, target=rgb_target, reduction='none')
        rgb_loss = rgb_loss.sum(1)  # sum over RGB channels
        return rgb_loss

    @staticmethod
    def compute_loss_rgb_lpips(pred, batch_data):
        from source.base.metrics import lpips
        rgb_target = batch_data['rgb_gt'].clone()
        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = pred[unknown_mask]
        rgb_loss = lpips(prediction=pred, target=rgb_target, net_type='alex')

        # broadcast to input B,H,W size
        b = rgb_loss.shape[0]
        h = rgb_target.shape[2]
        w = rgb_target.shape[3]
        rgb_loss = rgb_loss[:, None, None].broadcast_to((b, h, w))
        return rgb_loss

    @staticmethod
    def compute_loss_rgb_ssim(pred, batch_data):
        from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
        from torchvision.transforms.functional import resize
        rgb_target = batch_data['rgb_gt'].clone()
        unknown_mask = torch.isnan(rgb_target)
        rgb_target[unknown_mask] = pred[unknown_mask]
        pred_resized = resize(pred, size=[256, 256], antialias=True)
        rgb_target_resized = resize(rgb_target, size=[256, 256], antialias=True)
        rgb_loss = 1.0 - ms_ssim(pred_resized, rgb_target_resized, data_range=1.0, size_average=False, win_size=5)

        # broadcast to input B,H,W size
        b = rgb_loss.shape[0]
        h = rgb_target.shape[2]
        w = rgb_target.shape[3]
        rgb_loss = rgb_loss[:, None, None].broadcast_to((b, h, w))
        return rgb_loss
    
    @staticmethod
    def compute_loss_rgb_gradient(pred, batch_data):
        from source.base.metrics import gradient_loss_masked
        rgb_target = batch_data['rgb_gt'].clone()
        rgb_gradient_loss = gradient_loss_masked(pred, rgb_target)
        rgb_gradient_loss = torch.clip(rgb_gradient_loss, min=0.0, max=1.0)
        rgb_gradient_loss = rgb_gradient_loss.sum(1)  # sum over RGB channels
        return rgb_gradient_loss
    
    @staticmethod
    def compute_loss_rgb_fft(pred, batch_data):
        from source.base.metrics import fft_amplitude_loss
        rgb_loss = fft_amplitude_loss(pred, batch_data['rgb_gt'].clone())
        rgb_loss = torch.mean(rgb_loss, dim=1)  # mean over RGB channels
        return rgb_loss

    @staticmethod
    def slice_center_rgb(img: torch.Tensor, res_out: int) -> torch.Tensor:
        res_in = img.shape[2]  # assume square and channels first
        diff = (res_in - res_out) // 2
        img  = img[:, :, diff:diff + res_out, diff:diff + res_out]
        return img
    
    def compute_loss(self, pred, batch_data):
        import math
        from source.base.metrics import learned_loss_weighting, density_weighted_loss

        # keep same order as in yaml
        loss_tensor, loss_components_mean, loss_components = super().compute_loss(pred[:, 0], batch_data)

        if not self.has_color_output:
            return loss_tensor, loss_components_mean, loss_components
        
        new_loss_components = [
            # self.compute_loss_rgb_sparse(pred[:, 1:4], batch_data),
            IpesRgbd.compute_loss_rgb(pred[:, 1:4], batch_data),
            # IpesRgbd.compute_loss_rgb_huber(pred[:, 1:4], batch_data),
            # IpesRgbd.compute_loss_rgb_l1(pred[:, 1:4], batch_data),
            # IpesRgbd.compute_loss_rgb_lpips(pred[:, 1:4], batch_data),
            # IpesRgbd.compute_loss_rgb_ssim(pred[:, 1:4], batch_data),
            # IpesRgbd.compute_loss_rgb_gradient(pred[:, 1:4], batch_data),
            IpesRgbd.compute_loss_rgb_fft(pred[:, 1:4], batch_data),
        ]
        
        # density weighted loss
        rgb_target = batch_data['rgb_gt']
        valid_mask_rgb = ~torch.isnan(rgb_target[:, 0])
        # rgb_lin_center = self.slice_center_rgb(batch_data['patch_rgb_linear'], res_out=pred.shape[2])
        # rgb_nn_center = self.slice_center_rgb(batch_data['patch_rgb_nearest'], res_out=pred.shape[2])
        # new_loss_components[0] = density_weighted_loss(new_loss_components[0], rgb_lin_center, rgb_nn_center, alpha=5.0)
        
        loss_components = torch.cat((loss_components, torch.stack(new_loss_components)))

        valid_count_rgb = valid_mask_rgb.sum() + 1e-8
        new_loss_components_mean = torch.stack([torch.sum(loss) / valid_count_rgb for loss in new_loss_components])
        loss_components_mean = torch.cat((loss_components_mean, new_loss_components_mean))
        
        # learned weighting for RGB
        new_loss_components_mean_weighted = torch.zeros_like(new_loss_components_mean)
        new_loss_components_mean_weighted[0] = learned_loss_weighting(new_loss_components_mean[0], self.rgb_loss_weight[0])
        new_loss_components_mean_weighted[1] = learned_loss_weighting(new_loss_components_mean[1], self.rgb_fft_loss_weight[0])
        # new_loss_components_mean_weighted[1] = learned_loss_weighting(new_loss_components_mean[1], self.rgb_grad_loss_weight[0])
        loss_tensor = loss_tensor + new_loss_components_mean_weighted.sum()
        
        if math.isclose(loss_tensor.item(), 0.0):
            print('loss is close to zero')

        if math.isnan(loss_tensor.item()):
            print('loss is nan')

        return loss_tensor, loss_components_mean, loss_components

    def calc_metrics(self, pred, batch):
        hm_metrics = super().calc_metrics(pred, batch)

        if not self.has_color_output:
            return hm_metrics
        
        pred = pred.detach()
        pred_proc = self.post_proc_pred(batch, pred)
        pred_rgb = pred_proc[:, 1:4].detach()
        pred_rgb_flat = pred_rgb.flatten()
        
        rgb_target = batch['rgb_gt'].detach()
        rgb_target_flat = rgb_target.flatten()

        # ignore all nans (cut from tensor)
        rgb_target_nan = torch.isnan(rgb_target)
        rgb_pred_nan = torch.isnan(pred_rgb)
        rgb_nan = torch.logical_or(rgb_target_nan, rgb_pred_nan)
        rgb_nan_flat = rgb_nan.flatten()
        pred_rgb_flat_no_nan = pred_rgb_flat[~rgb_nan_flat]
        rgb_target_flat_no_nan = rgb_target_flat[~rgb_nan_flat]

        rgb_e = pred_rgb_flat_no_nan - rgb_target_flat_no_nan
        rgb_rmse = torch.sqrt(torch.mean(torch.square(rgb_e)))

        from source.base.metrics import psnr, lpips, gradient_rmse
        rgb_psnr = psnr(pred_rgb_flat_no_nan, rgb_target_flat_no_nan, 1.0)
        rgb_gradient_rmse = gradient_rmse(pred_rgb, rgb_target)

        # ignore all nans (fill from prediction)
        # pred_rgb = pred_proc[:, 1:4].detach()
        # rgb_target = batch['rgb_gt'].detach()
        # rgb_target_nan = torch.isnan(rgb_target)
        # rgb_target_no_nan = rgb_target.clone()
        # rgb_target_no_nan[rgb_target_nan] = pred_rgb[rgb_target_nan]
        # rgb_lpips = lpips(pred_rgb, rgb_target_no_nan)

        rgb_metrics = {
            'rgb_rmse': rgb_rmse,
            'rgb_psnr': rgb_psnr,
            'rgb_gradient_rmse': rgb_gradient_rmse,
            # 'rgb_lpips': rgb_lpips.mean(),
        }
        eval_dict = {**hm_metrics, **rgb_metrics}
        return eval_dict

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        
        self.log('epoch/val/weights/rgb_loss_weight', self.rgb_loss_weight.item(),
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        self.log('epoch/val/weights/rgb_fft_loss_weight', self.rgb_fft_loss_weight.item(),
                 on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        # self.log('epoch/val/weights/rgb_grad_loss_weight', self.rgb_grad_loss_weight.item(),
        #          on_step=False, on_epoch=True, logger=True, batch_size=batch['pts_query_ms'].shape[0])
        
        return loss

    def on_test_epoch_end(self):

        from source.base.evaluation import make_test_report
        from source.base.container import aggregate_dicts, flatten_dicts
        from source.dataloaders.base_data_module import read_shape_list, get_results_dir

        shape_names = read_shape_list(self.in_file)
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)

        outputs_flat = flatten_dicts(self.test_step_outputs)
        metrics_dicts_stacked = aggregate_dicts(outputs_flat, method='stack')

        test_set_file_name = os.path.basename(self.in_file)
        output_file = os.path.join(results_dir, 'metrics_{}_{}.xlsx'.format(self.name, test_set_file_name))
        metrics_keys_to_log = ('hm_rmse_ms', )
        if self.has_color_output:
            metrics_keys_to_log += ('rgb_rmse', 'rgb_psnr', 'rgb_gradient_rmse')
        low_metrics_better = [True, False, True, True, False, False, False]
        loss_total_mean, metrics = make_test_report(
            shape_names=shape_names, results=metrics_dicts_stacked,
            output_file=output_file, output_names=self.output_names, is_dict=True,
            metrics_keys_to_log=metrics_keys_to_log, low_metrics_better=low_metrics_better)

        hm_rmse_ms_mean = metrics[metrics_keys_to_log.index('hm_rmse_ms')]
        self.log('epoch/test/RMSE_ms', hm_rmse_ms_mean, on_step=False, on_epoch=True, logger=True)
        log_str = f'\nTest results (mean): Loss={loss_total_mean}, HM RMSE_ms={hm_rmse_ms_mean}'
        if self.has_color_output:
            rgb_psnr_mean = metrics[metrics_keys_to_log.index('rgb_psnr')]
            rgb_gradient_rmse_mean = metrics[metrics_keys_to_log.index('rgb_gradient_rmse')]
            log_str += f', RGB PSNR={rgb_psnr_mean}, RGB Gradient RMSE={rgb_gradient_rmse_mean}'
        print(log_str)

    def post_proc_pred(self, batch: dict, pred):
        if not self.has_color_output:
            return super().post_proc_pred(batch, pred[:, 0])[:, None]

        pred_hm = pred[:, 0]
        pred_rgb = pred[:, 1:4]

        pred_hm_post = super().post_proc_pred(batch, pred_hm)
        pred_post = torch.cat((pred_hm_post[:, None], pred_rgb), dim=1)

        return pred_post

    def visualize_step_results(self, batch_data: dict, predictions, losses, metrics, iteration: int, step: str):
        from source.base.visualization import images_to_figure

        if step == 'train' and not self.debug:
            return  # no visualization

        super().visualize_step_results(batch_data, predictions, losses, metrics, iteration, step)

        if not self.has_color_input and not self.has_color_output:
            return

        from source.dataloaders.base_data_module import get_results_dir
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        results_dir = os.path.join(results_dir, step)

        vis_params = get_vis_params(batch_data, step)
        vis_batches_range, hm_finite, norm_min, norm_max = vis_params

        fig_io_imgs = []

        # input RGB maps
        if self.has_color_input:
            # input maps as points
            # for k in batch_data.keys():
            #     if k.startswith('patch_rgb_'):
            #         k_pts = k.replace('patch_rgb_', 'patch_hm_')
            #         save_hm_as_pts(
            #             name='rgb_' + k + '_pts', step=step, iteration=iteration,
            #             hm_tensor_ps=batch_data[k_pts], batch_data=batch_data, results_dir=results_dir,
            #             vis_batches_range=vis_batches_range, color_factor=0.10,
            #             patch_radius=batch_data['patch_radius_interp_ms'][0].item(),
            #             colors=batch_data[k].detach().cpu().numpy().transpose(0, 2, 3, 1))

            for k in batch_data.keys():
                if k.startswith('patch_rgb_'):
                    rgb_pts = batch_data[k].detach().cpu().numpy().transpose(0, 2, 3, 1)
                    rgb_input_imgs = save_img_batch(
                        batch_data=batch_data, arr=rgb_pts, name='rgb_' + k, step=step, iteration=iteration,
                        results_dir=results_dir, vis_batches_range=vis_batches_range,
                        norm_min=0.0, norm_max=1.0)
                    fig_io_imgs.append(rgb_input_imgs)

        # prediction
        if self.has_color_output:
            # pred_hm_np = predictions.detach().cpu().numpy().astype(np.float32)[:, 0]  # done in super
            pred_rgb_np = predictions.detach().cpu().numpy().astype(np.float32)[:, 1:4].transpose(0, 2, 3, 1)

            pred_rgb_imgs = save_img_batch(
                batch_data=batch_data, arr=pred_rgb_np, name='rgb_pred',
                step=step, iteration=iteration,
                results_dir=results_dir, vis_batches_range=vis_batches_range,
                norm_min=0.0, norm_max=1.0)
            fig_io_imgs.append(pred_rgb_imgs)

            save_hm_as_pts(
                name='rgb_pred_pts', step=step, iteration=iteration,
                hm_tensor_ps=predictions[:, 0], batch_data=batch_data, results_dir=results_dir,
                vis_batches_range=vis_batches_range, color_factor=0.5,
                patch_radius=batch_data['patch_radius_hm_ms'][0].item(), colors=pred_rgb_np)

            # target and loss
            if step != 'predict':
                rgb_gt = batch_data['rgb_gt'].detach().cpu().numpy().transpose(0, 2, 3, 1)
                save_hm_as_pts(
                    name='rgb_gt_pts', step=step, iteration=iteration,
                    hm_tensor_ps=batch_data['hm_gt_ps'], batch_data=batch_data, results_dir=results_dir,
                    vis_batches_range=vis_batches_range, color_factor=0.25,
                    patch_radius=batch_data['patch_radius_hm_ms'][0].item(), colors=rgb_gt)

                rgb_gt_imgs = save_img_batch(
                    batch_data=batch_data, arr=rgb_gt,
                    name='rgb_gt', step=step, iteration=iteration,
                    results_dir=results_dir, vis_batches_range=vis_batches_range,
                    norm_min=0.0, norm_max=1.0)
                fig_io_imgs.append(rgb_gt_imgs)

                losses_np = losses[1].detach().cpu().numpy().astype(np.float32)
                loss_imgs = save_img_batch(
                    batch_data=batch_data, arr=losses_np, name='rgb_loss',
                    step=step, iteration=iteration,
                    results_dir=results_dir, vis_batches_range=vis_batches_range,
                    norm_min=0.0, norm_max=1.0)

                # make figures
                fig_loss_imgs = list(zip(pred_rgb_imgs, rgb_gt_imgs, loss_imgs))
                fig_loss = [p.replace('rgb_gt', '0_rgb_fig_loss') for p in rgb_gt_imgs]
                for img_tuple, fig_img in zip(fig_loss_imgs, fig_loss):
                    images_to_figure(img_tuple, fig_img)

                fig_inputs_imgs = list(zip(*fig_io_imgs))
                fig_inputs = [p.replace('rgb_pred', '0_rgb_fig_input') for p in pred_rgb_imgs]
                for img_tuple, fig_img in zip(fig_inputs_imgs, fig_inputs):
                    images_to_figure(img_tuple, fig_img)

                # merge hm and rgb figures
                fig_hm_loss = [p.replace('0_rgb_fig_loss', '0_hm_fig_loss') for p in fig_loss]
                fig_loss_figs_imgs = list(zip(*[fig_hm_loss, fig_loss]))
                fig_loss_figs = [p.replace('0_rgb_fig_loss', '00_hm_rgb_fig_loss') for p in fig_loss]
                for img_tuple, fig_img in zip(fig_loss_figs_imgs, fig_loss_figs):
                    images_to_figure(img_tuple, fig_img, img_width=None, img_height=None, concat_dim_x=False)

                fig_hm_input = [p.replace('0_rgb_fig_input', '0_hm_fig_input') for p in fig_inputs]
                fig_input_figs_imgs = list(zip(*[fig_hm_input, fig_inputs]))
                fig_input_figs = [p.replace('0_rgb_fig_input', '00_hm_rgb_fig_input') for p in fig_inputs]
                for img_tuple, fig_img in zip(fig_input_figs_imgs, fig_input_figs):
                    images_to_figure(img_tuple, fig_img, img_width=None, img_height=None, concat_dim_x=False)