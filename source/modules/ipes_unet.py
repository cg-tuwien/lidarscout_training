import typing

import torch
import pytorch_lightning as pl
from overrides import override

from source.modules.ipes_rgbd import IpesRgbd
from source.modules.ipes_cnn import IpesCnnNetwork
from source.modules.unet import UNet


class IpesUNet(IpesRgbd):

    def __init__(self,
                 hm_interp_size: int,
                 pts_to_img_methods: typing.List[str],
                 output_names,
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name):

        if not has_color_input or not has_color_output:
            raise ValueError('UNet requires color input and output')

        self.network_latent_size = network_latent_size
        self.in_channels = 4
        self.out_channels = 4
        self.hm_size = hm_size
        self.hm_interp_size = hm_interp_size

        self.in_file = in_file
        self.input_methods = pts_to_img_methods
        self.output_names = output_names
        self.results_dir = results_dir
        self.workers = workers

        super().__init__(has_color_input=has_color_input, has_color_output=has_color_output,
                         predict_batch_size=predict_batch_size,
                         debug=debug, show_unused_params=show_unused_params, name=name)

    @override
    def make_regressor(self):
        return IpesNetworkUNet(
            self.input_methods, self.in_channels, self.out_channels, self.hm_size)


class IpesNetworkUNet(pl.LightningModule):

    def __init__(self,
                 input_methods: typing.Collection[str],
                 input_channels, output_channels, hm_size):
        super(IpesNetworkUNet, self).__init__()

        self.input_methods = input_methods
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hm_size = hm_size

        self.unet = UNet(
            num_input_channels=self.input_channels*2,
            num_output_channels=self.output_channels*2,
            conv_block='gated',
        )

    @staticmethod
    def _replace_nan_(tensor: torch.Tensor):
        nan_mask = torch.isnan(tensor)
        # tensor[nan_mask] = torch.rand_like(tensor)[nan_mask]  # uniform noise
        tensor[nan_mask] = 0.0  # mean val
        return tensor

    def forward(self, batch):
        # network uses query points for batch dim -> need to flatten shape * query points dim
        hm_inputs = [batch['patch_hm_{}'.format(method)] for method in self.input_methods]
        rgb_inputs = [batch['patch_rgb_{}'.format(method)] for method in self.input_methods]

        b, _, h, w = hm_inputs[0].shape  # [b, r, r]

        inputs = torch.cat(hm_inputs + rgb_inputs, dim=1)

        # normalize HM to 0..10
        inputs[:, :len(hm_inputs)] = inputs[:, :len(hm_inputs)] * 5.0 + 5.0

        # replace RGB NaN with something, will get zero gradient
        inputs = self._replace_nan_(inputs) # [:, len(hm_inputs):])

        pred_res = self.unet(inputs)

        # de-normalize HM to -10..10
        pred_res = pred_res * 20.0 - 10.0
        # pred_res[:, 0] = pred_res[:, 0] * 20.0 - 10.0  # gradient error: inplace operation

        # slice center
        h_offset = (h - self.hm_size) // 2
        w_offset = (w - self.hm_size) // 2
        pred_res = pred_res[:, :, h_offset:h_offset + self.hm_size, w_offset:w_offset + self.hm_size]

        return pred_res  # [b, c, hres, hres]
