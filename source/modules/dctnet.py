# adapted from DCTNet:
# https://github.com/Zhaozixiang1228/GDSR-DCTNet/blob/main/model.py

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import pytorch_lightning as pl


class WeightPredictionNetwork(pl.LightningModule):
    def __init__(self, n_feats=64):
        super(WeightPredictionNetwork, self).__init__()
        feat = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, feat, kernel_size=1)
        self.conv_f = nn.Conv2d(feat, feat, kernel_size=1)
        self.conv_max = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat, feat, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feat, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(feat, feat, kernel_size=3, padding=1,
                                       stride=3, dilation=2)

    def forward(self, x):  # x is the input feature
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        x = f.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = f.interpolate(x, (short_cut.size(2), short_cut.size(3)),
                          mode='bilinear', align_corners=False)  # [b,f/4,14,14] -> [b,f/4,96,96]
        short_cut = self.conv_f(short_cut)
        x = self.conv4(x + short_cut)
        x = self.sigmoid(x)
        return x


class WeightPredictionNetworkSized(pl.LightningModule):
    def __init__(self, n_feats=64, h_input=96, w_input=96, h_target=64, w_target=64):
        super(WeightPredictionNetworkSized, self).__init__()
        self.h_input = h_input
        self.w_input = w_input
        self.h_target = h_target
        self.w_target = w_target

        feat = int(math.ceil(n_feats / 4))
        self.conv1 = nn.Conv2d(n_feats, feat, kernel_size=1)
        self.conv_f = nn.Conv2d(feat, feat, kernel_size=1)
        self.conv_max = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat, feat, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        # (((res - conv2.kernel - 1) // conv2.stride) - max_pool2d.kernel) // max_pool2d.stride
        fc_h = int(np.ceil((((self.h_input - 2) / 2) - 6) / 3))  # 14 for res=96
        fc_w = int(np.ceil((((self.w_input - 2) / 2) - 6) / 3))  # 14 for res=96
        fc_size = fc_h * fc_w  # * feat # same layer for every feature channel
        self.conv_fc = nn.Linear(fc_size, fc_size)
        self.conv3_ = nn.Conv2d(feat, feat, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feat, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(feat, feat, kernel_size=3, padding=1, stride=3, dilation=2)

    def forward(self, x):  # x is the input feature
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        x = f.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))

        # FC for global context
        shape_2d = x.shape
        x = self.relu(self.conv_fc(x.view(shape_2d[0] * shape_2d[1], -1))).view(*shape_2d)

        x = self.conv3_(x)
        x = f.interpolate(x, (self.h_target, self.w_target),
                          mode='bilinear', align_corners=False)  # [b,f/4,14,14] -> [b,f/4,96,96]
        short_cut = self.conv_f(short_cut)
        x = self.conv4(x + short_cut)
        x = self.sigmoid(x)
        return x


class CoupledLayer(pl.LightningModule):
    def __init__(self,
                 coupled_number=32,
                 n_feats=64,
                 kernel_size=3):
        super(CoupledLayer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_dlr, feat_rgb):
        short_cut = feat_dlr
        feat_dlr = f.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                            padding=1)
        feat_dlr = f.relu(feat_dlr, inplace=True)
        feat_dlr = f.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                            padding=1)
        feat_dlr = f.relu(feat_dlr + short_cut, inplace=True)
        short_cut = feat_rgb
        feat_rgb = f.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                            padding=1)
        feat_rgb = f.relu(feat_rgb, inplace=True)
        feat_rgb = f.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                            padding=1)
        feat_rgb = f.relu(feat_rgb + short_cut, inplace=True)
        return feat_dlr, feat_rgb


class CoupledEncoder(pl.LightningModule):
    def __init__(self,
                 n_feat=64,
                 n_layer=4,
                 n_channel=3,
                 n_channel_deep=1):
        super(CoupledEncoder, self).__init__()
        self.n_layer = n_layer
        self.init_deep = nn.Sequential(
            nn.Conv2d(n_channel_deep, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.init_rgb = nn.Sequential(
            nn.Conv2d(n_channel, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.coupled_feat_extractor = nn.ModuleList([CoupledLayer(n_feats=n_feat) for _ in range(self.n_layer)])

    def forward(self, feat_dlr, feat_rgb):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        for layer in self.coupled_feat_extractor:
            feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
        return feat_dlr, feat_rgb


class DecoderDeep(pl.LightningModule):
    def __init__(self, n_feats=64, n_channel_out=1, activation: nn.Module = nn.ReLU(True)):
        super(DecoderDeep, self).__init__()
        self.Decoder_Deep = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            activation,
            nn.Conv2d(n_feats // 2, n_feats // 4, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            activation,
            nn.Conv2d(n_feats // 4, n_channel_out, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            activation,
        )

    def forward(self, x):
        return self.Decoder_Deep(x)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().view(-1, n)  # loses batch dimension

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    v_c = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v = v_c[:, :, 0] * w_r - v_c[:, :, 1] * w_i

    if norm == 'ortho':
        v[:, 0] /= np.sqrt(n) * 2
        v[:, 1:] /= np.sqrt(n / 2) * 2

    v = 2 * v.view(*x_shape)

    return v


def idct(x, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    n = x_shape[-1]

    x_v = x.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        x_v[:, 0] *= np.sqrt(n) * 2
        x_v[:, 1:] *= np.sqrt(n / 2) * 2

    k = torch.arange(x_shape[-1], dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_r = x_v
    v_t_i = torch.cat([x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1)

    v_r = v_t_r * w_r - v_t_i * w_i
    v_i = v_t_r * w_i + v_t_i * w_r

    v = torch.view_as_real(torch.fft.ifft(torch.complex(v_r, v_i), dim=1))[..., 0]
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :n - (n // 2)]
    x[:, 1::2] += v.flip([1])[:, :n // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = dct(x, norm=norm)
    x2 = dct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def idct_2d(x, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(x, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def get_laplacian():
    from kornia.filters import Laplacian
    laplacian = Laplacian(3)
    return laplacian


def get_k(h: int, w: int):
    pi = torch.acos(torch.Tensor([-1]))
    cos_row = torch.cos(pi * torch.linspace(0, h - 1, h) / h).unsqueeze(1).expand(-1, w)
    cos_col = torch.cos(pi * torch.linspace(0, w - 1, w) / w).unsqueeze(0).expand(h, -1)
    kappa = 2 * (cos_row + cos_col - 2)
    return kappa[None, None, :, :]  # shape [1,1,h_target,w_target]


class DCTNet(pl.LightningModule):
    """
    Solver for the problem: min_{x} |x-d|_2^2+lambd|L(x)-L(r).*w_target|_2^2
    d - input low-resolution image
    r - guidance image (we want transfer the gradient of r into d)
        input RGB image
    z - output super-resolution image
    L - Laplacian operator
    w_target - Edge weight matrix (to be learned by WeightLearning Network)
        *Note: the solution of this problem is idct(p/c)
               p = dct(lambd*L(L(r)).*w_target + d)
               c = lambd*K^2+1
               K = self.get_K()
    """

    def __init__(self, n_feats=64, n_layer=4, n_channel=3, n_channel_deep=1):
        super(DCTNet, self).__init__()

        self.n_feats = n_feats
        self.lapl = get_laplacian()

        self.lambd = nn.Parameter(torch.nn.init.normal_(torch.zeros(size=(1, self.n_feats, 1, 1)), mean=0.1, std=0.3))
        self.wpnet = WeightPredictionNetwork(n_feats=n_feats)

        self.encoder_coupled = CoupledEncoder(n_feat=n_feats, n_layer=n_layer,
                                              n_channel=n_channel, n_channel_deep=n_channel_deep)
        self.decoder_depth = DecoderDeep(n_feats=n_feats, activation=nn.Tanh())

    def forward(self, x, y):
        # x - input depth image d, shape [n,c,h,w]
        # y - guidance RGB image r, shape [n,1,h,w] or [n,c,h,w]

        if len(y.shape) == 3:
            y = y[:, None, :, :]
        n, c, h, w = x.shape

        high_dim_d, high_dim_r = self.encoder_coupled(x, y)  # Cost: 2.2 GB

        kappa2 = get_k(h, w).pow(2)
        kappa2 = kappa2.to(x.dtype).to(x.device)

        # get weight
        weight = self.wpnet(high_dim_r)

        # get SR image (64 channel)
        lambd = torch.exp(self.lambd).to(x.device)

        dct_2d_input = lambd * self.lapl(torch.mul(self.lapl(high_dim_r), weight)) + high_dim_d  # Cost: 2.7 GB
        p = dct_2d(dct_2d_input)  # cost: 2.6 GB
        c = lambd * kappa2 + 1
        z = idct_2d(p / c)  # cost: 4 GB

        sr_depth = self.decoder_depth(z)
        return sr_depth


class DCTNetIpes(pl.LightningModule):
    """
    Solver for the problem: min_{x} |x-d|_2^2+lambd|L(x)-L(r).*w_target|_2^2
    d - input low-resolution image
    r - guidance image (we want transfer the gradient of r into d)
        input RGB image
    z - output super-resolution image
    L - Laplacian operator
    w_target - Edge weight matrix (to be learned by WeightLearning Network)
        *Note: the solution of this problem is idct(p/c)
               p = dct(lambd*L(L(r)).*w_target + d)
               c = lambd*K^2+1
               K = self.get_K()
    """

    def __init__(self, n_feats=64, n_layer=4, n_channel=3, n_channel_deep=1, n_channel_out=1,
                 h_input=96, w_input=96, h_target=64, w_target=64):
        super(DCTNetIpes, self).__init__()

        self.n_feats = n_feats
        self.lapl = get_laplacian()

        self.h_input = h_input
        self.w_input = w_input
        self.h_target = h_target
        self.w_target = w_target

        self.kappa2 = get_k(self.h_input, self.w_input).pow(2)  # [1,1,h,w]
        self.kappa2 = self.kappa2.detach()  # sub-graph is const and can be re-used
        self.lambd = nn.Parameter(torch.nn.init.normal_(torch.zeros(size=(1, self.n_feats, 1, 1)), mean=0.0, std=0.5))

        self.wpnet = WeightPredictionNetworkSized(
            n_feats=n_feats, h_input=h_input, w_input=w_input, h_target=self.h_input, w_target=self.w_input)
        self.encoder_coupled = CoupledEncoder(n_channel=n_channel, n_layer=n_layer,
                                              n_feat=n_feats, n_channel_deep=n_channel_deep)
        self.wp_fc = WeightPredictionNetworkSized(
            n_feats=n_feats, h_input=h_input, w_input=w_input, h_target=self.h_input, w_target=self.w_input)
        self.decoder_depth = [DecoderDeep(n_feats=n_feats, n_channel_out=1, activation=nn.LeakyReLU())
                              for _ in range(n_channel_out)]
        self.decoder_depth = nn.ModuleList(self.decoder_depth)

    def forward(self, x, y):
        # x - input depth image d, shape [b,c,h,w]
        # y - guidance RGB image r, shape [b,1,h,w] or [b,c,h,w]

        self.kappa2 = self.kappa2.to(self.device)

        high_dim_dlr, high_dim_rgb = self.encoder_coupled(x, y)  # [b,f,h,w]

        # get weight
        weight = self.wpnet(high_dim_rgb)  # [b,f,h,w]

        # get SR image (64 channel)
        lambd = torch.exp(self.lambd)  # [1,f,1,1]

        # get SR image (64 channel)
        dct_2d_input = lambd * self.lapl(torch.mul(self.lapl(high_dim_rgb), weight)) + high_dim_dlr  # [b,f,h,w]
        p = dct_2d(dct_2d_input)  # [b,f,h,w]
        c = lambd * self.kappa2 + 1  # [1,f,h,w]
        z = idct_2d(p / c)  # [b,f,h,w]
        z = z + torch.mul(z, self.wp_fc(z))  # [b,f,h,w], like cheap self-attention
        sr_depths = [decoder(z) for decoder in self.decoder_depth]
        sr_depth = torch.cat(sr_depths, dim=1)  # [b,o,h,w]
        return sr_depth
