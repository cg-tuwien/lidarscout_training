import os.path
from typing import Any, Optional, Callable, Union

import numpy as np
import torch
from imageio.v2 import imwrite
from torch.optim.optimizer import Optimizer
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from source.modules.ipes_rgbd import IpesRgbd
from source.modules.unlearned_module import UnlearnedModule
from source.base.fs import make_dir_for_file
from source.base.point_cloud import np_to_las, las2dem_np


class IpesLas2Dem(IpesRgbd, UnlearnedModule):

    def __init__(self,
                 has_color_input, has_color_output,
                 predict_batch_size, debug, show_unused_params, name,
                 gen_subsample_manifold, output_names, in_file, results_dir, num_pts_local, hm_size):

        super().__init__(
            has_color_input=has_color_input, has_color_output=has_color_output, predict_batch_size=predict_batch_size,
            debug=debug, show_unused_params=show_unused_params, name=name)

        self.gen_subsample_manifold = gen_subsample_manifold
        self.output_names = output_names
        self.in_file = in_file
        self.results_dir = results_dir
        self.num_pts_local = num_pts_local
        self.hm_size = hm_size

        from source.dataloaders.base_data_module import get_results_dir
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        self.regressor = Las2DemRegressor(hm_size=self.hm_size, results_dir=results_dir)


class Las2DemRegressor(pl.LightningModule):

    def __init__(self, hm_size: int, results_dir: str):
        super().__init__()

        self.hm_size = hm_size
        self.results_dir = results_dir

    @staticmethod
    def las2dem(pc_file_in: str, results_dir: str,
                pts_local_ms: np.ndarray, pts_local_rgb: np.ndarray,
                hm_size: int, pts_query_ms: np.ndarray, meters_per_pixel: float,
                lastools_dir: str = 'C:\\Program Files\\LAStools\\bin\\') -> tuple:

        import os
        from source.base.img import slice_img_center

        pc_file_in_mod = pc_file_in.replace(',', '_')
        pc_las_file = os.path.join(results_dir, 'las', pc_file_in_mod + '.las')  # e.g. 'ca_13,0,10'
        hm_padded_bil_file = os.path.join(results_dir, 'bil', pc_file_in_mod + '.bil')
        rgb_padded_png_file = os.path.join(results_dir, 'png', pc_file_in_mod + '.png')

        np_to_las(pts_local_ms, pts_local_rgb, pc_las_file)

        hm_size_padded = hm_size + hm_size // 2
        hm_size_padded_ms = hm_size_padded * meters_per_pixel
        # hm_size_ms = hm_size * meters_per_pixel

        verbose = ''  # '-v'
        step_m = meters_per_pixel
        kill_m = 640  # kill: edges longer than this will be deleted
        lower_left_corner = pts_query_ms - hm_size_padded_ms / 2
        extra_args = \
            (f'-elevation '
             f'{verbose} -step {step_m} -kill {kill_m} '
             f' -ll {lower_left_corner[0]} {lower_left_corner[1]} '
             f'-ncols {hm_size_padded} -nrows {hm_size_padded}')

        hm_padded, rgb_padded = las2dem_np(pc_las_file, hm_padded_bil_file, rgb_padded_png_file,
                                           extra_args, lastools_dir)

        # cut center out of the image
        hm = slice_img_center(hm_padded, hm_size_padded, hm_size, channels_first=False)
        rgb = slice_img_center(rgb_padded, hm_size_padded, hm_size, channels_first=False)
        # make_dir_for_file(img_out)
        # imwrite(img_out, img_center)
        return hm, rgb

    def forward(self, data):
        #  shape_data.pop('pts_ms')  # raw point cloud is large
        #  shape_data.pop('pts_rgb')  # raw point cloud is large
        #  shape_data.pop('pts_local_ms')  # would be collate of variable length
        #  shape_data.pop('pts_local_ps')  # would be collate of variable length
        #  shape_data.pop('pts_local_rgb')  # would be collate of variable length
        #  'pc_file_in' = {list: 1} ['ca_13,0,10']
        #  'meters_per_pixel' = {Tensor: (1,)} tensor([10.])
        #  'numerical_stability_factor' = {Tensor: (1,)} tensor([10.])
        #  'pts_query_ms' = {Tensor: (10, 3)} tensor([[6.6641e+05, 3.9447e+06, 1.3860e+01],\n        [6.5752e+05, 3.9570e+06, 7.5383e+02],\n        [6.5562e+05, 3.9592e+06, 5...2037e+05, 3.9054e+06, 3.1851e+02],\n        [6.6158e+05, 3.9513e+06, 2.1524e+02],\n        [7.0732e+05, 3.9150e+06, 2.1230e+02]])
        #  'hm_gt_ms' = {Tensor: (10, 64, 64)} tensor([[[     nan,      nan,      nan,  ...,  17.6748,  17.7173,  18.0206],\n         [     nan,      nan,      nan,  ...,  18..., 267.6891,  ..., 288.6521, 290.4869, 292.4449],\n         [274.4337, 272.4171, 270.5348,  ..., 290.5862, 292.6357, 294.8512]]])
        #  'hm_gt_ps' = {Tensor: (10, 64, 64)} tensor([[[    nan,     nan,     nan,  ...,  0.0562,  0.0568,  0.0613],\n         [    nan,     nan,     nan,  ...,  0.0639,  0....2,  0.8379,  0.8160,  ...,  1.1248,  1.1518,  1.1806],\n         [ 0.9153,  0.8856,  0.8579,  ...,  1.1533,  1.1835,  1.2161]]])
        #  'rgb_gt' = {Tensor: (10, 3, 64, 64)} tensor([[[[   nan,    nan,    nan,  ..., 0.3492, 0.3503, 0.3501],\n          [   nan,    nan,    nan,  ..., 0.3756, 0.3676, 0.3...   [0.3936, 0.3983, 0.3976,  ..., 0.4814, 0.4814, 0.4822],\n          [0.3927, 0.3988, 0.4166,  ..., 0.4804, 0.4865, 0.4888]]]])
        #  'patch_radius_interp_ms' = {Tensor: (1,)} tensor([678.8225], dtype=torch.float64)
        #  'patch_radius_hm_ms' = {Tensor: (1,)} tensor([678.8225], dtype=torch.float64)
        #  'pts_local_ms_z_mean' = {Tensor: (10,)} tensor([ 13.3988, 708.4134, 126.1224,  29.5566, 722.6055,   9.5244, 264.5577,\n        337.0590, 201.7205, 215.8871])
        #  __len__ = {int} 10

        if len(data['pts_local_ms']) != 1:
            print('batch may only contain 1 patch. did you set batch_size and dataset_step?')

        pc_file_in = data['pc_file_in'][0]
        meters_per_pixel = data['meters_per_pixel'][0].item()
        pts_local_ms = np.array(data['pts_local_ms'][0][0])
        pts_local_rgb = np.array(data['pts_local_rgb'][0][0])
        pts_query_ms = np.array(data['pts_query_ms'][0])
        hm_size = self.hm_size

        hm, rgb = self.las2dem(
            pc_file_in, self.results_dir,
            pts_local_ms, pts_local_rgb,
            hm_size, pts_query_ms, meters_per_pixel)
        hm_tensor = torch.tensor(hm[None, None, :, :], dtype=torch.float32)
        rgb_tensor = torch.tensor(rgb.transpose(2, 0, 1)[None, :3, :, :], dtype=torch.float32)
        res_tensor = torch.concat((hm_tensor, rgb_tensor), dim=1)
        return res_tensor
