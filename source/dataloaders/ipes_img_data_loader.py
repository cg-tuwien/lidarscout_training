import os.path
import typing

from overrides import override
import numpy as np

from source.base.point_cloud import pts_to_img, pts_to_img_cached, img_cache_file_path, rasterize_pts, interpolate_patch
from source.dataloaders.ipes_data_loader import IpesDataModule, IpesDataset, SHAPES_WITHOUT_VALID_RGB, img_cache_key_prefix


class IpesImgDataModule(IpesDataModule):

    def __init__(self, pts_to_img_methods: typing.List[str], rgb_to_img_methods: typing.List[str],
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, workers, use_ddp, seed,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, debug: bool, batch_size: int,
                 reconstruction_chunk_size: int = 500):
        super(IpesImgDataModule, self).__init__(
              use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
              do_data_augmentation=do_data_augmentation, debug=debug, batch_size=batch_size,
              seed=seed, context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size,
              hm_size=hm_size, meters_per_pixel=meters_per_pixel, dataset_step=dataset_step,
              reconstruction_chunk_size=reconstruction_chunk_size)

        self.pts_to_img_methods = pts_to_img_methods
        self.rgb_to_img_methods = rgb_to_img_methods

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
            # reconstruction (predict) doesn't use the on-disk image-render cache: a point cloud
            # is only ever reconstructed once, so there's no reuse to amortize the cache-write
            # cost/footprint against.
            use_cache=not reconstruction,
            debug=self.debug,
            reconstruction_chunk_size=self.reconstruction_chunk_size,
        )
        return dataset


class IpesImgDataset(IpesDataset):

    def __init__(self, pts_to_img_methods: typing.List[str], rgb_to_img_methods: typing.List[str], use_cache: bool,
                 load_gt: bool,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, seed, use_ddp,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug,
                 reconstruction_chunk_size: int = 500):
        super(IpesImgDataset, self).__init__(
            load_gt=load_gt, in_file=in_file, seed=seed,
            use_ddp=use_ddp, patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation,
            context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size,
            hm_size=hm_size, meters_per_pixel=meters_per_pixel, dataset_step=dataset_step, debug=debug,
            reconstruction_chunk_size=reconstruction_chunk_size)

        self.pts_to_img_methods = pts_to_img_methods
        self.rgb_to_img_methods = rgb_to_img_methods
        self.use_cache = use_cache
        # set by add_shape_data, consumed by add_gt_data within the same __getitem__ call (see
        # _check_fully_cached) -- safe as instance state since one worker processes one item at
        # a time, never concurrently
        self._fast_path_info = None

    @staticmethod
    def get_keys_to_augment(prefix: str, shape_data: dict) -> typing.List[str]:
        keys_to_augment = [k for k in shape_data.keys() if k.startswith(prefix)]

        # use normal key for already augmented data
        keys_to_augment = [k for k in keys_to_augment if not k.endswith('_aug')]
        return keys_to_augment

    @override
    def augment_data(self, shape_data: dict) -> dict:
        # data augmentation random flip along any number of axes
        # keys_to_negate = ['pts_ms', 'pts_query_ms']
        keys_to_negate = ['pts_query_ms', 'sun_pos_xy']  # pts_query_ms only for batch size
        keys_to_negate_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_flip = self.get_keys_to_augment('patch_hm_', shape_data) + \
                       self.get_keys_to_augment('patch_rgb_', shape_data)
        if self.load_gt:
            keys_to_flip += ['hm_gt_ms', 'hm_gt_ps', 'rgb_gt']
        shape_data = self.augment_flip(shape_data, keys_to_negate, keys_to_negate_list, keys_to_flip)

        # data augmentation random rotation by any 90 degree increments around vertical axis
        keys_to_rotate_grid = self.get_keys_to_augment('patch_hm_', shape_data) + \
                        self.get_keys_to_augment('patch_rgb_', shape_data)
        if self.load_gt:
            keys_to_rotate_grid += ['hm_gt_ms', 'hm_gt_ps', 'rgb_gt']
        keys_to_rotate_pts = ['pts_local_ms', 'pts_local_ps', 'sun_pos_xy']
        shape_data = self.augment_rotate(shape_data, keys_to_rotate_grid=keys_to_rotate_grid, keys_to_rotate_pts=keys_to_rotate_pts)

        # random scaling of z -> distorts understanding of heights
        # keys_to_scale_comp = ['pts_ms', 'pts_query_ms']
        keys_to_scale_comp = ['pts_query_ms']
        keys_to_scale_comp_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_scale_whole = self.get_keys_to_augment('patch_hm_', shape_data)
        keys_to_scale_whole = [k for k in keys_to_scale_whole if not k.endswith('_mask')]
        if self.load_gt:
            keys_to_scale_whole += ['hm_gt_ms', 'hm_gt_ps']
        shape_data = self.augment_z_scale(shape_data, keys_to_scale_comp, keys_to_scale_comp_list, keys_to_scale_whole)
        return shape_data

    def _img_cache_dir(self) -> str:
        from source.dataloaders.base_data_module import get_dataset_dir
        return os.path.join(get_dataset_dir(self.in_file), 'img_cache')

    def _check_fully_cached(self, shape_id) -> typing.Optional[typing.Tuple[str, int, int, str]]:
        """
        Returns (shape_name, start_id, end_id, cache_key_prefix) if every rendered image this
        window group needs already exists on disk, else None. Only stat()s candidate cache
        files -- never touches the point cloud, the KDTree, or the local-points CSR index.

        This is the point of the whole fast path (see add_shape_data/_add_gt_data_cached_only):
        on a fully-precomputed dataset (the common case for every epoch after the first), the
        upstream local-point resolution work is provably wasted whenever this returns non-None,
        since its result is only ever fed into pts_to_img_cached, which -- thanks to the
        cache_key fix -- doesn't even look at it on a hit. Nothing previously short-circuited
        that upstream work though, so every window, every epoch, still paged in chunks of the
        (potentially many-GB) point cloud and local-points index just to throw the result away.
        That's what was keeping ca_13_cp100's ~11GB CSR index and point cloud resident
        (Shared_Clean) across every worker, even though nothing in a warm-cache epoch actually
        needs their content.

        Restricted to load_gt (fit/test only; reconstruction never uses img_cache), use_cache,
        no augmentation (augmentation needs raw local points for its geometric transforms), and
        non-debug (visualization needs raw local points too) -- outside those, behavior is
        unchanged from before this fast path existed.
        """
        if not (self.load_gt and self.use_cache and not self.do_data_augmentation and not bool(self.debug)):
            return None

        shape_name, start_id, end_id = self.shape_names[shape_id].split(',')
        start_id, end_id = int(start_id), int(end_id)
        cache_dir = self._img_cache_dir()
        cache_key_prefix = img_cache_key_prefix(self.in_file, shape_name)
        has_rgb = shape_name not in SHAPES_WITHOUT_VALID_RGB

        for i in range(start_id, end_id):
            for method in self.pts_to_img_methods:
                key = '{}_{}_hm_{}'.format(cache_key_prefix, i, self.meters_per_pixel)
                path = img_cache_file_path(key, self.hm_interp_size, method, self.context_radius_factor, cache_dir)
                if not os.path.exists(path):
                    return None
            if has_rgb:
                for method in self.rgb_to_img_methods:
                    key = '{}_{}_rgb_{}'.format(cache_key_prefix, i, self.meters_per_pixel)
                    path = img_cache_file_path(key, self.hm_interp_size, method, self.context_radius_factor, cache_dir)
                    if not os.path.exists(path):
                        return None
        return shape_name, start_id, end_id, cache_key_prefix

    @override
    def add_shape_data(self, shape_id, shape_data: dict) -> dict:
        self._fast_path_info = self._check_fully_cached(shape_id)
        if self._fast_path_info is not None:
            # every image this item needs is already rendered and cached -- skip reading the
            # point cloud entirely, nothing downstream needs the actual points (see
            # _add_gt_data_cached_only)
            shape_data['sun_pos_xy'] = np.array([0.0, -1.0], dtype=np.float32)
            shape_data['pc_file_in'] = self.shape_names[shape_id]
            shape_data['meters_per_pixel'] = self.meters_per_pixel
            shape_data['numerical_stability_factor'] = 10.0
            return shape_data
        return super(IpesImgDataset, self).add_shape_data(shape_id, shape_data)

    def _add_gt_data_cached_only(self, shape_id, shape_data: dict, fast_path_info) -> dict:
        shape_name, start_id, end_id, cache_key_prefix = fast_path_info
        # cheap: only reads the small, mmap'd cache_gt/*.npy files (heightmaps_query, GT hm/rgb)
        # -- no point cloud, no KDTree, regardless of how densely this shape's points are sampled
        shape_data = self._load_gt_data(shape_id, shape_data)

        n = end_id - start_id
        shape_data['sun_pos_xy'] = np.broadcast_to(shape_data['sun_pos_xy'][np.newaxis], (n, 2)).copy()
        shape_data['patch_radius_interp_ms'] = self._get_patch_radius_p2(hm_res=self.hm_interp_size)
        shape_data['patch_radius_hm_ms'] = self._get_patch_radius_p2(hm_res=self.hm_size)

        cache_dir = self._img_cache_dir()
        has_rgb = shape_name not in SHAPES_WITHOUT_VALID_RGB

        for method in self.pts_to_img_methods:
            method_channels = 20 if method == 'rast_pyramid_mmm' else 1
            buf = np.zeros((n, method_channels, self.hm_interp_size, self.hm_interp_size), dtype=np.float32)
            for i in range(n):
                key = '{}_{}_hm_{}'.format(cache_key_prefix, start_id + i, self.meters_per_pixel)
                path = img_cache_file_path(key, self.hm_interp_size, method, self.context_radius_factor, cache_dir)
                buf[i] = np.load(path)
            shape_data['patch_hm_{}'.format(method)] = buf

        buf_rgb_shape = (n, 3, self.hm_interp_size, self.hm_interp_size)
        for method in self.rgb_to_img_methods:
            if has_rgb:
                rgb_buf = np.zeros(buf_rgb_shape, dtype=np.float32)
                for i in range(n):
                    key = '{}_{}_rgb_{}'.format(cache_key_prefix, start_id + i, self.meters_per_pixel)
                    path = img_cache_file_path(key, self.hm_interp_size, method, self.context_radius_factor, cache_dir)
                    rgb_buf[i] = np.load(path)
            else:
                rgb_buf = np.full(buf_rgb_shape, np.nan, dtype=np.float32)
            shape_data['patch_rgb_{}'.format(method)] = rgb_buf

        return shape_data

    @override
    def add_gt_data(self, shape_id, shape_data: dict) -> dict:
        fast_path_info, self._fast_path_info = self._fast_path_info, None
        if fast_path_info is not None:
            return self._add_gt_data_cached_only(shape_id, shape_data, fast_path_info)
        shape_data = super(IpesImgDataset, self).add_gt_data(shape_id, shape_data)

        from source.dataloaders.base_data_module import get_dataset_dir
        cache_dir = os.path.join(get_dataset_dir(self.in_file), 'img_cache')

        # cache_key_prefix/query_abs_ids: cheap, stat-only identifiers set by
        # IpesDataset._load_gt_data/_make_rec_data (see img_cache_key_prefix's docstring) --
        # absent only for ad-hoc/legacy callers, in which case pts_to_img_cached falls back to
        # its slower content-hash key.
        cache_key_prefix = shape_data.get('img_cache_key_prefix')
        query_abs_ids = shape_data.get('query_abs_ids')

        def _cache_key(kind: str, i: int) -> typing.Optional[str]:
            if cache_key_prefix is None or query_abs_ids is None:
                return None
            # meters_per_pixel is part of the key even though resolution/context_radius_factor
            # (also part of it, appended inside pts_to_img_cached) are the only other config
            # values normally varied -- patch_radius (and therefore which points get resolved
            # for a given query id) also depends on meters_per_pixel, so a change there must
            # invalidate old entries too, not silently reuse them.
            return '{}_{}_{}_{}'.format(cache_key_prefix, int(query_abs_ids[i]), kind, self.meters_per_pixel)

        pts_to_img_with_cache = lambda pts_ps_xy, pts_data, resolution, method, cache_key=None: pts_to_img_cached(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method=method, cache_dir=cache_dir,
            context_radius_factor=self.context_radius_factor, cache_key=cache_key)
        pts_to_img_no_cache = lambda pts_ps_xy, pts_data, resolution, method, cache_key=None: pts_to_img(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method=method,
            context_radius_factor=self.context_radius_factor)

        pts_to_img_func = pts_to_img_with_cache if self.use_cache else pts_to_img_no_cache

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
                        resolution=self.hm_interp_size, method=method, cache_key=_cache_key('hm', i))
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
                            resolution=self.hm_interp_size, method=method, cache_key=_cache_key('rgb', i))
                    rgb_buffer[i] = pts_rgb_img

                shape_data['patch_rgb_{}'.format(method)] = rgb_buffer
            return shape_data

        # shape_data = _add_gt_data_from_cache(shape_data)
        shape_data = _add_hms(shape_data)
        shape_data = _add_rgb(shape_data)

        if not self.do_data_augmentation and not bool(self.debug):
            # Match the cache-only fast path's schema exactly whenever it could have applied
            # instead of this slow path (i.e. whenever augmentation/debug are both off, which is
            # the fast path's own eligibility condition -- see _check_fully_cached). Without
            # this, a DataLoader batch mixing a fully-cached item (fast path, no pts_local_*
            # keys) with a partially-cached one (slow path, has them) would either silently drop
            # the slow item's extra fields (if the fast item collates first) or crash with a
            # KeyError (if the slow item collates first) -- collate_fn_custom keys a batch off
            # its first item only. Neither field is actually consumed downstream in this regime
            # (they only feed augmentation's geometric transforms or debug visualization, both
            # gated off here already).
            for key in ('pts_local_ms', 'pts_local_ps', 'pts_local_rgb', 'pts_local_ms_z_mean'):
                shape_data.pop(key, None)

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
