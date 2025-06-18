from typing import Any, Optional, Callable, Union

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from source.modules.ipes_rgbd import IpesRgbd
from source.modules.unlearned_module import UnlearnedModule


class IpesInterp(IpesRgbd, UnlearnedModule):

    def __init__(self,
                 has_color_input, has_color_output,
                 predict_batch_size, debug, show_unused_params, name,
                 gen_subsample_manifold, output_names, in_file, results_dir, num_pts_local, hm_size,
                 pts_to_img_methods, rgb_to_img_methods):

        super().__init__(
            has_color_input=has_color_input, has_color_output=has_color_output, predict_batch_size=predict_batch_size,
            debug=debug, show_unused_params=show_unused_params, name=name)

        self.gen_subsample_manifold = gen_subsample_manifold
        self.output_names = output_names
        self.in_file = in_file
        self.results_dir = results_dir
        self.num_pts_local = num_pts_local
        self.hm_size = hm_size
        self.pts_to_img_methods = pts_to_img_methods
        self.rgb_to_img_methods = rgb_to_img_methods

        if len(self.pts_to_img_methods) != 1:
            raise ValueError('Only one pts_to_img_method is allowed for interpolation')
        if len(self.rgb_to_img_methods) != 1:
            raise ValueError('Only one rgb_to_img_method is allowed for interpolation')

        self.regressor = IpesInterpolator(
            pts_to_img_method=self.pts_to_img_methods[0],
            rgb_to_img_method=self.rgb_to_img_methods[0])


class IpesInterpolator(pl.LightningModule):

    def __init__(self, pts_to_img_method: str, rgb_to_img_method: str):
        super().__init__()

        self.pts_to_img_method = pts_to_img_method
        self.rgb_to_img_method = rgb_to_img_method

    def forward(self, data):
        result_list = [data['patch_hm_{}'.format(self.pts_to_img_method)],
                       data['patch_rgb_{}'.format(self.rgb_to_img_method)]]
        res_tensor = torch.concat(result_list, dim=1)
        return res_tensor
