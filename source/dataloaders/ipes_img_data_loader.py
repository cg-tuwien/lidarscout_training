import os.path
import typing

from overrides import override
import numpy as np

from source.base.point_cloud import pts_to_img, pts_to_img_cached, rasterize_pts, interpolate_patch
from source.dataloaders.ipes_data_loader import IpesDataModule, IpesDataset


class IpesImgDataModule(IpesDataModule):

    def __init__(self, pts_to_img_methods: typing.List[str], rgb_to_img_methods: typing.List[str],
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, workers, use_ddp, seed,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, debug: bool, batch_size: int):
        super(IpesImgDataModule, self).__init__(
              use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
              do_data_augmentation=do_data_augmentation, debug=debug, batch_size=batch_size,
              seed=seed, context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size,
              hm_size=hm_size, meters_per_pixel=meters_per_pixel, dataset_step=dataset_step)

        self.pts_to_img_methods = pts_to_img_methods
        self.rgb_to_img_methods = rgb_to_img_methods

        # clear pts_to_img cache on start, first epoch will be slow
        from source.dataloaders.base_data_module import get_dataset_dir
        cache_dir = os.path.join(get_dataset_dir(self.in_file), 'img_cache')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)

    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):

        if reconstruction:
            patches_per_shape = None
            do_data_augmentation = False
            load_gt = False
        else:
            patches_per_shape = self.patches_per_shape
            do_data_augmentation = do_data_augmentation
            load_gt = True

        dataset = IpesImgDataset(
            in_file=in_file,
            context_radius_factor=self.context_radius_factor,
            hm_interp_size=self.hm_interp_size,
            hm_size=self.hm_size,
            seed=self.seed,
            patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation,
            use_ddp=self.use_ddp,
            meters_per_pixel=self.meters_per_pixel,
            dataset_step=self.dataset_step,
            load_gt=load_gt,
            pts_to_img_methods=self.pts_to_img_methods,
            rgb_to_img_methods=self.rgb_to_img_methods,
            use_cache=not reconstruction,
            debug=self.debug,
        )
        return dataset


class IpesImgDataset(IpesDataset):

    def __init__(self, pts_to_img_methods: typing.List[str], rgb_to_img_methods: typing.List[str], use_cache: bool,
                 load_gt: bool,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, seed, use_ddp,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug):
        super(IpesImgDataset, self).__init__(
            load_gt=load_gt, in_file=in_file, seed=seed,
            use_ddp=use_ddp, patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation,
            context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size,
            hm_size=hm_size, meters_per_pixel=meters_per_pixel, dataset_step=dataset_step, debug=debug)

        self.pts_to_img_methods = pts_to_img_methods
        self.rgb_to_img_methods = rgb_to_img_methods
        self.use_cache = use_cache

    @staticmethod
    def get_keys_to_augment(prefix: str, shape_data: dict) -> typing.List[str]:
        keys_to_augment = [k for k in shape_data.keys() if k.startswith(prefix)]

        # use normal key for already augmented data
        keys_to_augment = [k for k in keys_to_augment if not k.endswith('_aug')]
        return keys_to_augment

    @override
    def augment_data(self, shape_data: dict) -> dict:
        # keys_to_negate = ['pts_ms', 'pts_query_ms']
        keys_to_negate = ['pts_query_ms']
        keys_to_negate_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_flip = self.get_keys_to_augment('patch_hm_', shape_data) + \
                       self.get_keys_to_augment('patch_rgb_', shape_data)
        if self.load_gt:
            keys_to_flip += ['hm_gt_ms', 'hm_gt_ps', 'rgb_gt']
        shape_data = self.augment_flip(shape_data, keys_to_negate, keys_to_negate_list, keys_to_flip)

        # keys_to_scale_comp = ['pts_ms', 'pts_query_ms']
        keys_to_scale_comp = ['pts_query_ms']
        keys_to_scale_comp_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_scale_whole = self.get_keys_to_augment('patch_hm_', shape_data)
        if self.load_gt:
            keys_to_scale_whole += ['hm_gt_ms', 'hm_gt_ps']
        shape_data = self.augment_z_scale(shape_data, keys_to_scale_comp, keys_to_scale_comp_list, keys_to_scale_whole)
        return shape_data

    @override
    def add_gt_data(self, shape_id, shape_data: dict) -> dict:
        shape_data = super(IpesImgDataset, self).add_gt_data(shape_id, shape_data)

        from source.dataloaders.base_data_module import get_dataset_dir
        cache_dir = os.path.join(get_dataset_dir(self.in_file), 'img_cache')
        pts_to_img_cached_with_cache_dir = lambda pts_ps_xy, pts_data, resolution, method: pts_to_img_cached(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method=method, cache_dir=cache_dir)
        pts_to_img_func = pts_to_img_cached_with_cache_dir if self.use_cache else pts_to_img

        # add hms
        def _add_hms(shape_data):
            pts_local_ps = shape_data['pts_local_ps']
            pts_query_ps = shape_data['pts_query_ms']
            for method in self.pts_to_img_methods:
                method_channels = 20 if method == 'rast_pyramid_mmm' else 1
                buffer_shape = (pts_query_ps.shape[0], method_channels, self.hm_interp_size, self.hm_interp_size)
                hm_buffer = np.zeros(buffer_shape, dtype=np.float32)

                for i, pts_ps in enumerate(pts_local_ps):
                    hm = pts_to_img_func(
                        pts_ps_xy=pts_ps[:, :2], pts_data=pts_ps[:, 2],
                        resolution=self.hm_interp_size, method=method)
                    hm_buffer[i] = hm

                shape_data['patch_hm_{}'.format(method)] = hm_buffer
            return shape_data

        # add RGB
        def _add_rgb(shape_data):
            pts_local_ps = shape_data['pts_local_ps']
            pts_query_ps = shape_data['pts_query_ms']
            pts_local_rgb = shape_data['pts_local_rgb']
            buffer_rgb_shape = (pts_query_ps.shape[0], 3, self.hm_interp_size, self.hm_interp_size)
            for method in self.rgb_to_img_methods:
                rgb_buffer = np.zeros(buffer_rgb_shape, dtype=np.float32)

                for i, pts_ps in enumerate(pts_local_ps):
                    pts_rgb_img = pts_to_img_func(
                        pts_ps_xy=pts_ps[:, :2], pts_data=pts_local_rgb[i],
                            resolution=self.hm_interp_size, method=method)
                    rgb_buffer[i] = pts_rgb_img

                shape_data['patch_rgb_{}'.format(method)] = rgb_buffer
            return shape_data

        # shape_data = _add_gt_data_from_cache(shape_data)
        shape_data = _add_hms(shape_data)
        shape_data = _add_rgb(shape_data)
        return shape_data


def _test_pts_to_img():
    import matplotlib.pyplot as plt

    # manual test data
    # pts_ps = np.array([[-0.5, -0.5, 0.0],
    #                    [0.5, 0.5, 1.0],
    #                    [0.0, 0.0, 0.5],
    #                    [1.0, 1.0, 0.5],
    #                    [-0.5, 0.5, 0.0],
    #                    [0.5, -0.5, 1.0]])

    # random test data (square)
    # pts_ps = np.random.rand(100, 3) * 2.0 - 1.0

    # random test data (circle)
    phase = np.random.rand(100) * 2.0 * np.pi
    dist = np.random.rand(100)
    pts_ps = np.zeros((100, 3))
    pts_ps[:, 0] = np.cos(phase) * dist
    pts_ps[:, 1] = np.sin(phase) * dist
    pts_ps[:, 2] = np.random.rand(100) * 2.0 - 1.0

    resolution = 70

    hm_rasterize = rasterize_pts(pts_ps[:, :2], pts_ps[:, 2], resolution)
    hm_rasterize[np.isnan(hm_rasterize)] = 0.0
    hm_nearest = interpolate_patch(
        pts_ps_xy=pts_ps[:, :2], pts_data=pts_ps[:, 2], resolution=resolution, method='nearest')
    hm_linear = interpolate_patch(
        pts_ps_xy=pts_ps[:, :2], pts_data=pts_ps[:, 2], resolution=resolution, method='linear')
    hm_cubic = interpolate_patch(
        pts_ps_xy=pts_ps[:, :2], pts_data=pts_ps[:, 2], resolution=resolution, method='cubic')

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(hm_rasterize)
    axs[0, 0].set_title('rasterize')
    axs[0, 1].imshow(hm_nearest)
    axs[0, 1].set_title('nearest')
    axs[1, 0].imshow(hm_linear)
    axs[1, 0].set_title('linear')
    axs[1, 1].imshow(hm_cubic)
    axs[1, 1].set_title('cubic')

    plt.show()


if __name__ == '__main__':
    _test_pts_to_img()
