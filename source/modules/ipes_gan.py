import typing
import types

import torch
from torch import nn
import pytorch_lightning as pl
from overrides import override

from source.modules.ipes_cnn import IpesCnn 

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # A tiny 3-layer CNN that grades 64x64 patches.
        # It outputs a grid of values (Real vs. Fake) rather than a single scalar.
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class IpesGan(IpesCnn):
    def __init__(self,
                 hm_interp_size: int,
                 pts_to_img_methods: typing.List[str],
                 output_names,
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name):

        # 1. Initialize the parent CNN class
        super().__init__(hm_interp_size, pts_to_img_methods, output_names, hm_size,
                         in_file, results_dir, network_latent_size, workers,
                         has_color_input, has_color_output,
                         predict_batch_size, debug, show_unused_params, name)
        
        if not self.has_color_output:
            raise RuntimeError('IpesGan requires has_color_output=True for full GAN training.')

        # GAN-specific components
        self.discriminator = PatchDiscriminator(in_channels=3)
        self.gan_loss = nn.BCEWithLogitsLoss()

        # MANDATORY for PyTorch Lightning GANs:
        # We must manually control opt.step() and opt.zero_grad().
        self.automatic_optimization = False

        # LightningCLI may inject a single-optimizer configure_optimizers; rebind
        # this instance to the GAN implementation so training always uses G and D.
        self.configure_optimizers = types.MethodType(IpesGan.configure_optimizers, self)

    @override
    def configure_optimizers(self):
        # opt_generator trains the IpesCnnNetwork (inherited from self.regressor)
        assert self.regressor is not None
        opt_generator = torch.optim.AdamW(self.regressor.parameters(), lr=1e-4)
        # opt_discriminator trains our Discriminator
        opt_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4)
        return [opt_generator, opt_discriminator], []

    @override
    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            raise RuntimeError('IpesGan requires exactly two optimizers (generator, discriminator).')
        if len(optimizers) != 2:
            raise RuntimeError(f'IpesGan expected 2 optimizers, got {len(optimizers)}.')

        opt_generator = optimizers[0]
        opt_discriminator = optimizers[1]

        # =========================
        # 1. TRAIN GENERATOR (IpesCnnNetwork)
        # =========================
        self.toggle_optimizer(opt_generator)

        # Run the standard forward pass to get the baseline L2 anchor
        loss_l2, loss_components_mean, loss_components, metrics_dict, pred = self.common_step(
            batch=batch, step='train')

        pred_rgb = pred[:, 1:4]
        gt_rgb = batch['rgb_gt']

        # Create a valid mask to handle colorless SWISSS3D patches safely
        valid_mask = ~torch.isnan(gt_rgb)
        gt_safe = torch.nan_to_num(gt_rgb, nan=0.0)
        pred_safe = pred_rgb * valid_mask.float()

        total_g_loss = loss_l2

        if valid_mask.any():
            # Adversarial Penalty: Generator wants Discriminator to output 1 (Real)
            fake_preds = self.discriminator(pred_safe)
            g_gan_loss = self.gan_loss(fake_preds, torch.ones_like(fake_preds))

            # Combine the base L2 anchor with the GAN sharpness penalty
            total_g_loss = loss_l2 + (0.1 * g_gan_loss)
            self.log('loss/train/g_gan_loss', g_gan_loss, prog_bar=True)

        # PyTorch Lightning manual backward API
        opt_generator.zero_grad()
        self.manual_backward(total_g_loss)
        opt_generator.step()

        self.untoggle_optimizer(opt_generator)

        # =========================
        # 2. TRAIN DISCRIMINATOR
        # =========================
        if valid_mask.any():
            self.toggle_optimizer(opt_discriminator)

            # Train Discriminator to output 1 for the Real Ground Truth
            real_preds = self.discriminator(gt_safe)
            d_real_loss = self.gan_loss(real_preds, torch.ones_like(real_preds))

            # Train Discriminator to output 0 for the Fake CNN prediction
            # .detach() is CRITICAL here so we don't backprop into the generator!
            fake_preds_d = self.discriminator(pred_safe.detach())
            d_fake_loss = self.gan_loss(fake_preds_d, torch.zeros_like(fake_preds_d))

            total_d_loss = (d_real_loss + d_fake_loss) / 2

            # PyTorch Lightning manual backward API
            opt_discriminator.zero_grad()
            self.manual_backward(total_d_loss)
            opt_discriminator.step()

            self.untoggle_optimizer(opt_discriminator)

            self.log('loss/train/d_loss', total_d_loss, prog_bar=True)

        # Log the standard metrics (RMSE, PSNR, etc.) using the existing pipeline
        self.do_logging(total_g_loss, loss_components_mean, log_type='train',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='hm_rmse_ms')
        return total_g_loss