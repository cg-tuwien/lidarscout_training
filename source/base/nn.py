import typing

import torch
from torch.nn import functional as f
import pytorch_lightning as pl
import pytorch_lightning.profilers


class BaseModule(pl.LightningModule):

    def __init__(self, debug, show_unused_params, name):
        super().__init__()

        self.debug = debug
        self.show_unused_params = show_unused_params
        self.name = name

    def on_after_backward(self):
        # for finding disconnected parts
        # DDP won't run by default if such parameters exist
        # find_unused_parameters makes it run but is slower
        if self.show_unused_params:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print('Unused param {}'.format(name))
            self.show_unused_params = False  # print once is enough

    def do_logging(self, loss_total, loss_components, log_type: str, output_names: list, metrics_dict: dict,
                   key_to_log_prog_bar: str,
                   keys_to_log=frozenset({'abs_dist_rms', 'accuracy', 'precision', 'recall', 'f1_score'}),
                   show_in_prog_bar=True, on_step=True, on_epoch=False):

        # import math

        self.log('loss/{}/00_all'.format(log_type), loss_total, on_step=on_step, on_epoch=on_epoch)
        for li, l in enumerate(loss_components):
            self.log('loss/{}/{}'.format(log_type, output_names[li]), l, on_step=on_step, on_epoch=on_epoch)

        for key in metrics_dict.keys():
            if key in keys_to_log:
                value = metrics_dict[key].item()
                # if math.isnan(value):
                #     value = 0.0
                self.log('metrics/{}/{}'.format(log_type, key), value, on_step=on_step, on_epoch=on_epoch)

        # only command line
        self.log('cmd/{}/{}'.format(log_type, key_to_log_prog_bar), metrics_dict[key_to_log_prog_bar],
                 on_step=on_step, on_epoch=on_epoch, logger=False, prog_bar=show_in_prog_bar)

    def get_prog_bar(self):
        from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
        prog_bar = self.trainer.progress_bar_callback
        if prog_bar is not None and not isinstance(prog_bar, TQDMProgressBar):
            print('Warning: invalid progress bar type: {}'.format(type(prog_bar)))
        else:
            prog_bar = typing.cast(typing.Optional[TQDMProgressBar], prog_bar)
        return prog_bar


# https://github.com/numpy/numpy/issues/5228
def cartesian_to_polar(pts_cart: torch.Tensor):
    batch_size = pts_cart.shape[0]
    num_pts = pts_cart.shape[1]
    num_dim = pts_cart.shape[2]
    pts_cart_flat = pts_cart.reshape((-1, num_dim))

    def pol_2d():
        x = pts_cart_flat[:, 0]
        y = pts_cart_flat[:, 1]

        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        return torch.stack((r, phi), dim=1)

    def pol_3d():
        x = pts_cart_flat[:, 0]
        y = pts_cart_flat[:, 1]
        z = pts_cart_flat[:, 2]

        hxy = torch.hypot(x, y)
        r = torch.hypot(hxy, z)
        el = torch.atan2(z, hxy)
        az = torch.atan2(y, x)
        return torch.stack((az, el, r), dim=1)

    pts_spherical_flat = pol_2d() if num_dim == 2 else pol_3d()
    pts_spherical = pts_spherical_flat.reshape((batch_size, num_pts, num_dim))

    return pts_spherical


def pos_encoding(pts: torch.Tensor, pos_encoding_levels: int, skip_last_dim=False):
    """
    use positional encoding on points
    3d example: [x, y, z] -> [f(cos, x), f(cos, y), f(cos, z), f(sin, x), f(sin, y), f(sin, z)]
    :param pts: tensor[b, n, 2 or 3]
    :param pos_encoding_levels: int
    :param skip_last_dim: bool, skip last dim of input points (necessary for radius of polar coordinates)
    :return:
    """

    if pos_encoding_levels <= 0:
        return pts

    batch_size = pts.shape[0]
    num_pts = pts.shape[1]
    num_dim = pts.shape[2]
    num_dim_out = num_dim * 2 * pos_encoding_levels
    pts_enc = torch.zeros((batch_size, num_pts, num_dim_out), device=pts.device)

    for dim in range(num_dim):
        for lvl in range(pos_encoding_levels):
            dim_out = dim * lvl * 2
            if skip_last_dim and dim == num_dim - 1:
                pts_enc[..., dim_out] = pts[..., dim]
                pts_enc[..., dim_out + num_dim] = pts[..., dim]
            else:
                pts_enc[..., dim_out] = torch.cos(pts[..., dim] * lvl * torch.pi * pow(2.0, lvl))
                pts_enc[..., dim_out + num_dim] = torch.sin(pts[..., dim] * lvl * torch.pi * pow(2.0, lvl))

    return pts_enc


class AttentionPoco(pl.LightningModule):
    # self-attention for feature vectors
    # adapted from POCO attention
    # https://github.com/valeoai/POCO/blob/4e39b5e722c82e91570df5f688e2c6e4870ffe65/networks/decoder/interp_attention.py

    def __init__(self, net_size_max=1024, reduce=True):
        super(AttentionPoco, self).__init__()

        self.fc_query = torch.nn.Conv2d(net_size_max, 1, 1)
        self.fc_value = torch.nn.Conv2d(net_size_max, net_size_max, 1)
        self.reduce = reduce

    def forward(self, feature_vectors: torch.Tensor):
        # [feat_len, batch, num_feat] expected -> feature dim to dim 0
        feature_vectors_t = torch.permute(feature_vectors, (1, 0, 2))

        query = self.fc_query(feature_vectors_t).squeeze(0)  # fc over feature dim -> [batch, num_feat]
        value = self.fc_value(feature_vectors_t).permute(1, 2, 0)  # -> [batch, num_feat, feat_len]

        weights = torch.nn.functional.softmax(query, dim=-1)  # softmax over num_feat -> [batch, num_feat]
        if self.reduce:
            feature_vector_out = torch.sum(value * weights.unsqueeze(-1).broadcast_to(value.shape), dim=1)
        else:
            feature_vector_out = (weights.unsqueeze(2) * value).permute(0, 2, 1)
        return feature_vector_out


def batch_quat_to_rotmat(q, out=None):
    """
    quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
    :param q:
    :param out:
    :return:
    """

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2 / torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


class STN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), self.dim*self.dim)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max * self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        batch_size = x.size()[0]
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), 4)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max*self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x_quat = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x_quat)

        return x, x_quat


class PointNetfeat(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500,
                 polar=False, use_point_stn=True, use_feat_stn=True,
                 output_size=100, sym_op='max', dim=3):
        super(PointNetfeat, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.num_scales = num_scales
        self.polar = polar
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.output_size = output_size
        self.dim = dim

        if self.use_point_stn:
            self.stn1 = QSTN(net_size_max=net_size_max, num_scales=self.num_scales,
                             num_points=num_points, dim=dim, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(net_size_max=net_size_max, num_scales=self.num_scales,
                            num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = torch.nn.BatchNorm1d(64)
        self.bn0b = torch.nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_size, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(output_size)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(output_size, output_size*self.num_scales, 1)
            self.bn4 = torch.nn.BatchNorm1d(output_size*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            pass
        elif self.sym_op == 'wsum':
            pass
        elif self.sym_op == 'att':
            self.att = AttentionPoco(output_size)
        else:
            raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

    def forward(self, x, pts_weights):

        # input transform
        if self.use_point_stn:
            trans, trans_quat = self.stn1(x[:, :3, :])  # transform only point data
            # an error here can mean that your input size is wrong (e.g. added normals in the point cloud files)
            x_transformed = torch.bmm(trans, x[:, :3, :])  # transform only point data
            x = torch.cat((x_transformed, x[:, 3:, :]), dim=1)
        else:
            trans = None
            trans_quat = None

        if bool(self.polar):
            x = torch.permute(x, (0, 2, 1))
            x = cartesian_to_polar(pts_cart=x)
            x = torch.permute(x, (0, 2, 1))

        # mlp (64,64)
        x = f.relu(self.bn0a(self.conv0a(x)))
        x = f.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = torch.bmm(trans2, x)
        else:
            trans2 = None

        # mlp (64,128,output_size)
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (output_size,output_size*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(f.relu(x)))

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'wsum':
                pts_weights_bc = torch.broadcast_to(torch.unsqueeze(pts_weights, 1), size=x.shape)
                x = x * pts_weights_bc
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'att':
                x = self.att(x)
            else:
                raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

        else:
            x_scales = x.new_empty(x.size(0), self.output_size*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)
            x = x_scales

        x = x.view(-1, self.output_size * self.num_scales ** 2)

        return x, trans, trans_quat, trans2


class MLP(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int, num_layers: int,
                 halving_size=True, final_bn_act=False, final_layer_norm=False,
                 activation: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 norm: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
                 fc_layer=torch.nn.Linear, dropout=0.0):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        if halving_size:
            layer_sizes = [int(input_size / (2 ** i)) for i in range(num_layers)]
        else:
            layer_sizes = [input_size for _ in range(num_layers)]

        fully_connected = [fc_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(num_layers-1)]
        norms = [norm(layer_sizes[i + 1]) for i in range(num_layers - 1)]

        layers_list = []
        for i in range(self.num_layers-1):
            layers_list.append(torch.nn.Sequential(
                fully_connected[i],
                norms[i],
                activation(),
                torch.nn.Dropout(dropout),
            ))

        final_modules = [fc_layer(layer_sizes[-1], output_size)]
        if final_bn_act:
            if final_layer_norm:
                final_modules.append(torch.nn.LayerNorm(output_size))
            else:
                final_modules.append(norm(output_size))
            final_modules.append(activation())
        final_layer = torch.nn.Sequential(*final_modules)
        layers_list.append(final_layer)

        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.layers.forward(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PPSProfiler(pytorch_lightning.profilers.PyTorchProfiler):
    from typing import Any, Optional, Union
    from pathlib import Path

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_module_names: bool = True,
        with_stack: bool = False,
        **profiler_kwargs: Any,
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename, group_by_input_shapes=group_by_input_shapes,
                         emit_nvtx=emit_nvtx, export_to_chrome=export_to_chrome, row_limit=row_limit,
                         sort_by_key=sort_by_key, record_module_names=record_module_names, with_stack=with_stack,
                         **profiler_kwargs)
