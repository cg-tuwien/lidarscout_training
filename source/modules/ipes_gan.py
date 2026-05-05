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
        # harvest intermediate features
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        feat1 = self.layer1(img)
        feat2 = self.layer2(feat1)
        out = self.layer3(feat2)
        
        # Return both the final logits and the intermediate feature maps
        return out, [feat1, feat2]

class PatchDiscriminator_4lvls(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Layer 1: 64x64 -> 32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2: 32x32 -> 16x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3: 16x16 -> 8x8 (The Macro-Structure Layer)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4: Final Output
        self.layer4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        feat1 = self.layer1(img)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        out = self.layer4(feat3)
        
        # Return final prediction AND the 3 intermediate feature maps
        return out, [feat1, feat2, feat3]

class IpesGan(IpesCnn):
    def __init__(self,
                 hm_interp_size: int,
                 pts_to_img_methods: typing.List[str],
                 output_names,
                 hm_size,
                 in_file, results_dir, network_latent_size, workers,
                 has_color_input: bool,
                 has_color_output: bool,
                 predict_batch_size, debug, show_unused_params, name,
                 loss_module: typing.Any = None, use_valid_pixel_mask: bool = False,
                 valid_pixel_mask_key: str = 'patch_hm_mask', use_sun_direction: bool = True,
                 discriminator_levels: int = 3, 
                 learned_gan_weights: bool = True,
                 gan_loss_weight: float = 0.01,
                 feature_matching_weight: float = 0.5,
                 train_metrics_every_n_steps: int = 1):

        # 1. Initialize the parent CNN class
        super().__init__(hm_interp_size, pts_to_img_methods, output_names, hm_size,
                         in_file, results_dir, network_latent_size, workers,
                         has_color_input, has_color_output,
                         predict_batch_size, debug, show_unused_params, name,
                         loss_module=loss_module, use_valid_pixel_mask=use_valid_pixel_mask,
                         valid_pixel_mask_key=valid_pixel_mask_key, use_sun_direction=use_sun_direction,
                         train_metrics_every_n_steps=train_metrics_every_n_steps)
        
        if not self.has_color_output:
            raise RuntimeError('IpesGan requires has_color_output=True for full GAN training.')

        # GAN-specific components
        if discriminator_levels == 3:
            self.discriminator = PatchDiscriminator(in_channels=3)
        elif discriminator_levels == 4:
            self.discriminator = PatchDiscriminator_4lvls(in_channels=3)
        else:
            raise ValueError(f'Unsupported discriminator_levels={discriminator_levels}')
        
        self.gan_loss = nn.BCEWithLogitsLoss()
        
        # =========================
        # DYNAMIC WEIGHTING SETUP
        # =========================
        self.learned_gan_weights = learned_gan_weights
        if self.learned_gan_weights:
            # Learnable variance parameters initialized to 0 (precision = 1.0)
            self.gan_log_var = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.feat_log_var = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            # Fallback to hardcoded weights
            self.gan_loss_weight = gan_loss_weight
            self.feature_matching_weight = feature_matching_weight

        # MANDATORY for PyTorch Lightning GANs:
        # We must manually control opt.step() and opt.zero_grad().
        self.automatic_optimization = False

        # LightningCLI may inject a single-optimizer configure_optimizers; rebind
        # this instance to the GAN implementation so training always uses G and D.
        self.configure_optimizers = types.MethodType(IpesGan.configure_optimizers, self)

        if self.loss_module is not None:
            component_names = getattr(self.loss_module, 'component_names', None)
            if component_names is not None:
                self.output_names = list(component_names)
            else:
                loss_name = getattr(self.loss_module, 'name', None)
                if loss_name is not None:
                    self.output_names = [loss_name]

    @override
    def configure_optimizers(self):
        # opt_generator trains the IpesCnnNetwork (inherited from self.regressor)
        assert self.regressor is not None
        
        # We must explicitly hand the new learnable GAN weight parameters 
        # to the Generator's optimizer so they update alongside the CNN layers!
        g_params = list(self.regressor.parameters())
        if self.learned_gan_weights:
            g_params.extend([self.gan_log_var, self.feat_log_var])
            
        opt_generator = torch.optim.AdamW(g_params, lr=1e-4)
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
        # 1. TRAIN GENERATOR
        # =========================
        self.toggle_optimizer(opt_generator)

        # Run the standard forward pass to get the baseline L2 anchor
        # NOTE: loss_l2 is already processed by the Uncertainty Weighting in losses.py
        loss_l2, loss_components_mean, loss_components, metrics_dict, pred = self.common_step(
            batch=batch, step='train', batch_idx=batch_idx)
        
        gt_rgb = batch['rgb_gt']
        gt_all = gt_rgb  # Isolated strictly to textures
        valid_mask = ~torch.isnan(gt_all)
        gt_safe = torch.nan_to_num(gt_all, nan=0.0)
        
        pred_safe = pred[:, 1:] * valid_mask.float()

        total_g_loss = loss_l2
        
        if valid_mask.any():
            with torch.no_grad():
                _, real_features = self.discriminator(gt_safe)

            fake_preds, fake_features = self.discriminator(pred_safe)
            g_gan_loss = self.gan_loss(fake_preds, torch.ones_like(fake_preds))

            feat_loss = 0.0
            for r_feat, f_feat in zip(real_features, fake_features):
                feat_loss += nn.functional.l1_loss(f_feat, r_feat.detach())

            # =========================
            # MATHEMATICAL MERGING
            # =========================
            if self.learned_gan_weights:
                # Calculate precision modifiers
                gan_precision = torch.exp(-self.gan_log_var)
                feat_precision = torch.exp(-self.feat_log_var)
                
                # Apply Uncertainty Weighting Math: (1/sigma^2) * Loss + log(sigma)
                weighted_gan_loss = gan_precision * g_gan_loss + self.gan_log_var
                weighted_feat_loss = feat_precision * feat_loss + self.feat_log_var
                
                total_g_loss = loss_l2 + weighted_gan_loss + weighted_feat_loss
                
                # Log the active weight multipliers so you can track what the network prioritizes
                self.log('loss/weights/gan_adversarial', gan_precision, prog_bar=False)
                self.log('loss/weights/gan_feature_match', feat_precision, prog_bar=False)
            else:
                total_g_loss = loss_l2 + (self.gan_loss_weight * g_gan_loss) + (self.feature_matching_weight * feat_loss)
            
            self.log('loss/train/g_gan_loss', g_gan_loss, prog_bar=True)
            self.log('loss/train/g_feat_loss', feat_loss, prog_bar=True)

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

            real_preds, _ = self.discriminator(gt_safe)
            d_real_loss = self.gan_loss(real_preds, torch.ones_like(real_preds))

            fake_preds_d, _ = self.discriminator(pred_safe.detach())
            d_fake_loss = self.gan_loss(fake_preds_d, torch.zeros_like(fake_preds_d))

            total_d_loss = (d_real_loss + d_fake_loss) / 2

            opt_discriminator.zero_grad()
            self.manual_backward(total_d_loss)
            opt_discriminator.step()
            self.untoggle_optimizer(opt_discriminator)

            self.log('loss/train/d_loss', total_d_loss, prog_bar=True)

        self.do_logging(total_g_loss, loss_components_mean, log_type='train',
                        output_names=self.output_names, metrics_dict=metrics_dict, show_in_prog_bar=True,
                        keys_to_log=self.keys_to_log, key_to_log_prog_bar='hm_rmse_ms',
                        log_metrics=bool(metrics_dict))
        return total_g_loss
