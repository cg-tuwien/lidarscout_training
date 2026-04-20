import typing

import torch
from torch import nn
import pytorch_lightning as pl
from overrides import override

from source.modules.ipes_rgbd import IpesRgbd


class IpesCnn(IpesRgbd):

    def __init__(self,
                 hm_interp_size: int,
                 pts_to_img_methods: typing.List[str],
                 output_names,
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name,
                 loss_module=None, use_valid_pixel_mask: bool = False,
                 valid_pixel_mask_key: str = 'patch_hm_mask', use_sun_direction: bool = True,
                 train_metrics_every_n_steps: int = 1):

        self.network_latent_size: int = network_latent_size
        self.hm_size: int = hm_size
        self.hm_interp_size: int = hm_interp_size

        self.in_file: str = in_file
        self.input_methods = pts_to_img_methods
        self.output_names = list(output_names)
        self.results_dir: str = results_dir
        self.workers: int = workers
        self.use_sun_direction = use_sun_direction

        super().__init__(has_color_input=has_color_input, has_color_output=has_color_output,
                         predict_batch_size=predict_batch_size, 
                         debug=debug, show_unused_params=show_unused_params, name=name,
                         loss_module=loss_module, use_valid_pixel_mask=use_valid_pixel_mask,
                         valid_pixel_mask_key=valid_pixel_mask_key,
                         train_metrics_every_n_steps=train_metrics_every_n_steps)

        if self.loss_module is not None:
            component_names = getattr(self.loss_module, 'component_names', None)
            if component_names is not None:
                self.output_names = list(component_names)
            else:
                loss_name = getattr(self.loss_module, 'name', None)
                if loss_name is not None:
                    self.output_names = [loss_name]

    @override
    def make_regressor(self):
        return IpesCnnNetwork(
            input_methods=self.input_methods,
            latent_size=self.network_latent_size,
            hm_interp_size=self.hm_interp_size,
            hm_size=self.hm_size,
            has_color_input=self.has_color_input,
            has_color_output=self.has_color_output,
            use_sun_direction=self.use_sun_direction,
        )
        
    def on_train_epoch_end(self):
        # step GAN schedulers manually
        schedulers = self.lr_schedulers()
        if schedulers:
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]
            for sch in schedulers:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(sch, ReduceLROnPlateau):
                    continue
                sch.step()


class IpesCnnNetwork(pl.LightningModule):

    def __init__(self, 
                 input_methods: typing.Collection[str],
                 latent_size, hm_interp_size, hm_size, has_color_input, has_color_output,
                 use_sun_direction: bool = True):
        super(IpesCnnNetwork, self).__init__()

        self.input_methods = input_methods
        self.num_input_methods = len(input_methods)
        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.has_color_input = has_color_input
        self.has_color_output = has_color_output
        self.use_sun_direction = use_sun_direction

        self.input_channels = 4 if has_color_input else 1
        self.output_channels = 4 if has_color_output else 1

        def _make_encoder(in_channels, out_channels):
            # gets 96x96, returns 1x1, hand-tuned sizes
            return nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, kernel_size=3, stride=3),
            )

        def _make_decoder(in_channels, out_channels):
            # gets 1x1, returns 96x96, hand-tuned sizes
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, 64, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, out_channels, kernel_size=8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=9),
            )

        # HM encoder
        self.hm_encoders = [_make_encoder(in_channels=1, out_channels=latent_size)
                            for _ in range(self.num_input_methods)]
        self.hm_encoders = nn.ModuleList(self.hm_encoders)
        self.hm_merger = nn.Sequential(
            nn.Conv2d(latent_size * self.num_input_methods, latent_size * self.num_input_methods * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # RGB encoder
        if self.has_color_input:
            self.rgb_encoders = [_make_encoder(in_channels=3, out_channels=latent_size)
                                for _ in range(self.num_input_methods)]
            self.rgb_encoders = nn.ModuleList(self.rgb_encoders)
            self.rgb_merger = nn.Sequential(
                nn.Conv2d(latent_size * self.num_input_methods, latent_size * self.num_input_methods * 2, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        # HM-RGB merger
        hm_rgb_merger_output_size = latent_size * 4 * self.num_input_methods \
            if self.has_color_input else latent_size * 2 * self.num_input_methods
        self.hm_rgb_merger = nn.Sequential(
            nn.Conv2d(hm_rgb_merger_output_size, hm_rgb_merger_output_size // 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Decoder per channel
        self.decoders = [_make_decoder(in_channels=hm_rgb_merger_output_size // 2, out_channels=1)
                         for _ in range(self.output_channels)]
        self.decoders = nn.ModuleList(self.decoders)

        # Residual
        # residual_in_channels = self.output_channels + self.input_channels * self.num_input_methods + 1
        residual_in_channels = self.output_channels + self.input_channels * self.num_input_methods
        if self.use_sun_direction:
            residual_in_channels += 2  # for sun pos 2D vector
        self.residual = nn.Sequential(
            nn.Conv2d(residual_in_channels, residual_in_channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_in_channels, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.output_channels, kernel_size=9, padding=4),
        )

    @staticmethod
    def _replace_nan_(tensor: torch.Tensor, replace_val=0.5):
        nan_mask = torch.isnan(tensor)
        # tensor[nan_mask] = torch.rand_like(tensor)[nan_mask]  # uniform noise
        tensor[nan_mask] = replace_val  # mean val

    def forward(self, batch):
        # network uses query points for batch dim -> need to flatten shape * query points dim
        hm_inputs = [batch['patch_hm_{}'.format(method)] for method in self.input_methods]
        rgb_inputs = []
        sun_pos_xy = batch['sun_pos_xy']

        # replace RGB NaN with noise, will get zero gradient
        for hm_input in hm_inputs:
            self._replace_nan_(hm_input, 0.0)

        if self.has_color_input:
            rgb_inputs = [batch['patch_rgb_{}'.format(method)] for method in self.input_methods]

            # replace RGB NaN with noise, will get zero gradient
            for rgb_input in rgb_inputs:
                self._replace_nan_(rgb_input)

        b, _, h, w = hm_inputs[0].shape  # [b, r, r]

        # interpolated patches are normalized with a different resolution than the output patches
        hm_inputs = [
            hm_input if method == 'mask' else hm_input / self.hm_size * self.hm_interp_size
            for method, hm_input in zip(self.input_methods, hm_inputs)
        ]

        # # debug with GT
        # hm_pts_flat = batch['hm_gt_ps'].view(hm_input.shape)
        # hm_pts_flat[torch.isnan(hm_pts_flat)] = 0.0

        # HM
        hm_inputs_enc = [encoder(hm_input) for encoder, hm_input in zip(self.hm_encoders, hm_inputs)]
        hm_inputs_enc = torch.cat(hm_inputs_enc, dim=1)
        hm_inputs_merged = self.hm_merger(hm_inputs_enc)

        # RGB
        if self.has_color_input:
            rgb_inputs_enc = [encoder(rgb_input) for encoder, rgb_input in zip(self.rgb_encoders, rgb_inputs)]
            rgb_inputs_enc = torch.cat(rgb_inputs_enc, dim=1)
            rgb_inputs_merged = self.rgb_merger(rgb_inputs_enc)
        else:
            rgb_inputs_merged = torch.zeros((hm_inputs_merged.shape[0],) + (0,) + hm_inputs_merged.shape[2:],
                                            device=hm_inputs_merged.device)

        # HM-RGB merger
        hm_rgb_inputs = torch.cat((hm_inputs_merged, rgb_inputs_merged), dim=1)
        hm_rgb_inputs = self.hm_rgb_merger(hm_rgb_inputs)

        # Decoder
        pred = [decoder(hm_rgb_inputs) for decoder in self.decoders]
        pred = torch.cat(pred, dim=1)

        # Residual, slice xy middle to target resolution
        if self.has_color_input:
            all_inputs = torch.cat(hm_inputs + rgb_inputs, dim=1)
        else:
            all_inputs = torch.cat(hm_inputs, dim=1)
        if self.use_sun_direction:
            sun_dir_map = sun_pos_xy[:, :, None, None].expand(-1, -1, h, w)
            all_inputs = torch.cat((all_inputs, sun_dir_map), dim=1)
        pred = torch.cat((pred, all_inputs), dim=1)
        pred_res = self.residual(pred)
        h_offset = (h - self.hm_size) // 2
        w_offset = (w - self.hm_size) // 2
        pred_res = pred_res[:, :, h_offset:h_offset + self.hm_size, w_offset:w_offset + self.hm_size]

        return pred_res  # [b, c, hres, hres]
