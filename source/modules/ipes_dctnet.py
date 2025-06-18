import torch
import pytorch_lightning as pl
from overrides import override

from source.modules.ipes_rgbd import IpesRgbd


class IpesDCTNet(IpesRgbd):

    def __init__(self,
                 hm_interp_size: int,
                 use_mlp: bool,
                 mlp_layers, dropout,
                 output_names,
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name):

        self.use_mlp = use_mlp
        self.mlp_layers = mlp_layers
        self.network_latent_size = network_latent_size
        self.dropout = dropout
        self.in_channels = 4 if has_color_input else 1
        self.out_channels = 4 if has_color_output else 1
        self.hm_size = hm_size
        self.hm_interp_size = hm_interp_size

        self.in_file = in_file
        self.output_names = output_names
        self.results_dir = results_dir
        self.workers = workers

        super().__init__(has_color_input=has_color_input, has_color_output=has_color_output,
                         predict_batch_size=predict_batch_size,
                         debug=debug, show_unused_params=show_unused_params, name=name)

    @override
    def make_regressor(self):
        return IpesNetworkDCTNet(
            latent_size=self.network_latent_size,
            hm_interp_size=self.hm_interp_size,
            hm_size=self.hm_size,
            dct_input_channels=self.in_channels,
            dct_output_channels=self.out_channels,
            use_mlp=self.use_mlp,
        )


class IpesNetworkDCTNet(pl.LightningModule):

    def __init__(self, latent_size, hm_interp_size, hm_size,
                 dct_input_channels=1, dct_output_channels=1, use_mlp=False):
        super(IpesNetworkDCTNet, self).__init__()

        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.dct_input_channels = dct_input_channels
        self.dct_output_channels = dct_output_channels
        self.use_mlp = use_mlp

        self.dct_output_interp_size = True
        # self.dct_output_interp_size = False

        # DCTNet
        from source.modules.dctnet import DCTNetIpes
        self.network = DCTNetIpes(n_feats=latent_size, n_layer=4,
                                  n_channel=self.dct_input_channels, n_channel_deep=self.dct_input_channels,
                                  n_channel_out=self.dct_output_channels,
                                  h_input=hm_interp_size, w_input=hm_interp_size,
                                  h_target=hm_size, w_target=hm_size,
                                  )

        # MLP
        if self.use_mlp:
            from source.base.nn import MLP
            if self.dct_output_interp_size:
                mlp_input_size = hm_interp_size**2
            else:
                mlp_input_size = hm_size**2
            self.mlp = MLP(input_size=mlp_input_size,
                           output_size=hm_size**2,  # * self.dct_output_channels,
                           num_layers=1, halving_size=False)
            self.mlp_rgb = MLP(input_size=mlp_input_size,
                               output_size=hm_size**2,
                               num_layers=1, halving_size=False)
            # self.mlp = [MLP(input_size=mlp_input_size, output_size=hm_size**2,
            #                 num_layers=1, halving_size=False)
            #             for _ in range(self.dct_output_channels)]
            # self.mlp = nn.ModuleList(self.mlp)

    @staticmethod
    def _replace_nan_(tensor: torch.Tensor):
        nan_mask = torch.isnan(tensor)
        # tensor[nan_mask] = torch.rand_like(tensor)[nan_mask]  # uniform noise
        tensor[nan_mask] = 0.5  # mean val

    def forward(self, batch):
        # network uses query points for batch dim -> need to flatten shape * query points dim
        # for some reason, NN is better for guidance than linear
        hm_input = torch.concat((batch['patch_hm_linear'], batch['patch_rgb_linear']), dim=1)
        hm_guidance = torch.concat((batch['patch_hm_nearest'], batch['patch_rgb_nearest']), dim=1)
        b, c, h, w = hm_input.shape  # [b, c, r, r]
        hm_shape = (b, c, self.hm_size, self.hm_size)

        # # debug with GT
        # hm_pts_flat = batch['hm_gt_ps'].view(hm_input.shape)
        # hm_pts_flat[torch.isnan(hm_pts_flat)] = 0.0

        # replace RGB NaN with noise, will get zero gradient
        self._replace_nan_(hm_input[:, 1:])
        self._replace_nan_(hm_guidance[:, 1:])

        # DCTNet
        hm_input[:, 0] = hm_input[:, 0] * 0.5 + 0.5  # normalize HM to 0..1
        hm_guidance[:, 0] = hm_guidance[:, 0] * 0.5 + 0.5  # normalize HM to 0..1
        pred_hm = self.network.forward(hm_input, hm_guidance)
        pred_hm[:, 0] = pred_hm[:, 0] * 2.0 - 1.0  # de-normalize HM to -1..1

        # extra MLP
        if self.use_mlp:
            if self.dct_output_interp_size:
                hm_interp_vector_shape_flat = (b, self.dct_output_channels, self.hm_interp_size**2)
            else:
                hm_interp_vector_shape_flat = (b, self.dct_output_channels, self.hm_size**2)
            pred_hm_flat = torch.reshape(pred_hm, hm_interp_vector_shape_flat)

            # pred_hm = self.mlp.forward(pred_hm_flat)  # single channel
            # pred_hms = [mlp(pred_hm_flat[:, i]) for i, mlp in enumerate(self.mlp)]
            pred_hms = [self.mlp(pred_hm_flat[:, 0])] + \
                       [self.mlp_rgb(pred_hm_flat[:, i]) for i in range(1, self.dct_output_channels)]
            pred_hm = torch.cat(pred_hms, dim=1)  # [b,o,h,w]
            pred_hm = pred_hm.view(hm_shape)
        else:
            # slice xy middle to target resolution
            if pred_hm.shape[-2:] != (self.hm_size, self.hm_size):
                h, w = pred_hm.shape[-2:]
                h_offset = (h - self.hm_size) // 2
                w_offset = (w - self.hm_size) // 2
                pred_hm = pred_hm[:, :, h_offset:h_offset + self.hm_size,
                                  w_offset:w_offset + self.hm_size]

        return pred_hm  # [b, q, c, hres, hres]
