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
                 output_names: typing.Optional[typing.Sequence[str]],
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name,
                 loss_module: typing.Any = None, use_valid_pixel_mask: bool = False,
                 valid_pixel_mask_key: str = 'patch_hm_mask', use_sun_direction: bool = False,
                 train_metrics_every_n_steps: int = 1):

        self.network_latent_size: int = network_latent_size
        self.hm_size: int = hm_size
        self.hm_interp_size: int = hm_interp_size

        self.in_file: str = in_file
        self.input_methods = pts_to_img_methods
        self.output_names = list(output_names) if output_names is not None else []
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

        if len(self.output_names) == 0:
            raise ValueError(
                'output_names is required unless it can be derived from model.init_args.loss_module '
                '(component_names or name).'
            )

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

        # ==========================================
        # 1. THE HEIGHTMAP STREAM (Geometry Path)
        # ==========================================
        self.hm_encoders = [_make_encoder(in_channels=1, out_channels=latent_size)
                            for _ in range(self.num_input_methods)]
        self.hm_encoders = nn.ModuleList(self.hm_encoders)
        self.hm_merger = nn.Sequential(
            nn.Conv2d(latent_size * self.num_input_methods, latent_size * self.num_input_methods * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.hm_decoder = _make_decoder(in_channels=latent_size * self.num_input_methods * 2, out_channels=1)
            
        hm_residual_in = 1 + self.num_input_methods + 1  # hm_pred + hm_inputs + mask
        if self.use_sun_direction:
            hm_residual_in += 2
            
        self.hm_residual = nn.Sequential(
            nn.Conv2d(hm_residual_in, hm_residual_in, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hm_residual_in, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=9, padding=4),
        )

        # ==========================================
        # 2. THE RGB STREAM (Texture Path)
        # ==========================================
        if self.has_color_output:
            if self.has_color_input:
                self.rgb_encoders = [_make_encoder(in_channels=3, out_channels=latent_size)
                                    for _ in range(self.num_input_methods)]
                self.rgb_encoders = nn.ModuleList(self.rgb_encoders)
                self.rgb_merger = nn.Sequential(
                    nn.Conv2d(latent_size * self.num_input_methods, latent_size * self.num_input_methods * 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                rgb_decoder_in = latent_size * self.num_input_methods * 4 # 2x from RGB, 2x from detached HM
                rgb_residual_in = 4 + (4 * self.num_input_methods) + 1 # pred_rgb(3) + pred_hm(1) + rgb_in(3*N) + hm_in(1*N) + mask(1*N)
            else:
                rgb_decoder_in = latent_size * self.num_input_methods * 2 # only gets detached HM features
                rgb_residual_in = 4 + (1 * self.num_input_methods) + 1 # pred_rgb(3) + pred_hm(1) + hm_in(1*N) + mask(1*N)

            self.rgb_decoder = _make_decoder(in_channels=rgb_decoder_in, out_channels=3)

            if self.use_sun_direction:
                rgb_residual_in += 2

            self.rgb_residual = nn.Sequential(
                nn.Conv2d(rgb_residual_in, rgb_residual_in, kernel_size=9, padding=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(rgb_residual_in, 32, kernel_size=9, padding=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=9, padding=4),
            )

    @staticmethod
    def _replace_nan_(tensor: torch.Tensor, replace_val=0.5):
        nan_mask = torch.isnan(tensor)
        tensor[nan_mask] = replace_val  

    def forward(self, batch):
        hm_inputs = [batch['patch_hm_{}'.format(method)] for method in self.input_methods]
        rgb_inputs = []
        sun_pos_xy = batch['sun_pos_xy']
        
        # Extract and prepare the global 96x96 mask for the forward pass
        # We use a default fallback of ones if the mask isn't present in a dataset
        if 'patch_hm_mask' in batch:
            fwd_mask = batch['patch_hm_mask'].to(dtype=hm_inputs[0].dtype, device=hm_inputs[0].device)
            if fwd_mask.ndim == 3:
                fwd_mask = fwd_mask.unsqueeze(1)
        else:
            fwd_mask = torch.ones_like(hm_inputs[0])

        for hm_input in hm_inputs:
            self._replace_nan_(hm_input, 0.0)

        if self.has_color_input:
            rgb_inputs = [batch['patch_rgb_{}'.format(method)] for method in self.input_methods]
            for rgb_input in rgb_inputs:
                self._replace_nan_(rgb_input)

        b, _, h, w = hm_inputs[0].shape 

        hm_inputs = [
            hm_input if method == 'mask' else hm_input / self.hm_size * self.hm_interp_size
            for method, hm_input in zip(self.input_methods, hm_inputs)
        ]

        if self.use_sun_direction:
            sun_dir_map = sun_pos_xy[:, :, None, None].expand(-1, -1, h, w)

        # ==========================================
        # 1. PROCESS HEIGHTMAP
        # ==========================================
        hm_inputs_enc = [encoder(hm_input) for encoder, hm_input in zip(self.hm_encoders, hm_inputs)]
        hm_inputs_enc = torch.cat(hm_inputs_enc, dim=1)
        hm_inputs_merged = self.hm_merger(hm_inputs_enc)

        pred_hm_base = self.hm_decoder(hm_inputs_merged)

        hm_res_inputs = [pred_hm_base] + hm_inputs + [fwd_mask]
        if self.use_sun_direction:
            hm_res_inputs.append(sun_dir_map)
        
        hm_res_input_tensor = torch.cat(hm_res_inputs, dim=1)
        pred_hm_res = self.hm_residual(hm_res_input_tensor)

        h_offset = (h - self.hm_size) // 2
        w_offset = (w - self.hm_size) // 2
        pred_hm_final = pred_hm_res[:, :, h_offset:h_offset + self.hm_size, w_offset:w_offset + self.hm_size]

        if not self.has_color_output:
            return pred_hm_final

        # ==========================================
        # 2. PROCESS RGB (The One-Way Bridge)
        # ==========================================
        if self.has_color_input:
            rgb_inputs_enc = [encoder(rgb_input) for encoder, rgb_input in zip(self.rgb_encoders, rgb_inputs)]
            rgb_inputs_enc = torch.cat(rgb_inputs_enc, dim=1)
            rgb_inputs_merged = self.rgb_merger(rgb_inputs_enc)
            
            # The Bridge: RGB sees HM features, but gradients stop here
            rgb_decoder_input = torch.cat((rgb_inputs_merged, hm_inputs_merged.detach()), dim=1)
        else:
            rgb_decoder_input = hm_inputs_merged.detach()

        pred_rgb_base = self.rgb_decoder(rgb_decoder_input)

        # The Bridge: RGB residual sees HM predictions, but gradients stop here
        if self.has_color_input:
            rgb_res_inputs = [pred_hm_base.detach(), pred_rgb_base] + hm_inputs + rgb_inputs + [fwd_mask]
        else:
            rgb_res_inputs = [pred_hm_base.detach(), pred_rgb_base] + hm_inputs + [fwd_mask]
            
        if self.use_sun_direction:
            rgb_res_inputs.append(sun_dir_map)

        rgb_res_input_tensor = torch.cat(rgb_res_inputs, dim=1)
        pred_rgb_res = self.rgb_residual(rgb_res_input_tensor)

        pred_rgb_final = pred_rgb_res[:, :, h_offset:h_offset + self.hm_size, w_offset:w_offset + self.hm_size]

        # ==========================================
        # 3. COMBINE OUTPUT
        # ==========================================
        # Output format [B, 4, H, W] expected by losses.py
        return torch.cat((pred_hm_final, pred_rgb_final), dim=1)