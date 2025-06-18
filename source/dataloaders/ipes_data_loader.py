import os.path
import typing

from overrides import override
import numpy as np

from source.dataloaders.base_data_module import BaseDataModule, BaseDataset
from source.base.math import hm_to_pts
from source.base.normalization import hm_model_space_to_patch_space, hm_patch_space_to_model_space


class IpesDataModule(BaseDataModule):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 seed, in_file, workers, use_ddp,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, debug: bool, batch_size: int):
        super(IpesDataModule, self).__init__(
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size, debug=debug, seed=seed)

        self.context_radius_factor = context_radius_factor
        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.meters_per_pixel = meters_per_pixel
        self.dataset_step = dataset_step

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

        dataset = IpesDataset(
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
            debug=self.debug,
        )
        return dataset


class IpesDataset(BaseDataset):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, seed, use_ddp, load_gt: bool,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug):

        self.dataset_step = dataset_step
        self.shape_names_raw = []

        super(IpesDataset, self).__init__(
            in_file=in_file, seed=seed, use_ddp=use_ddp,
            patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation,
            load_gt=load_gt, debug=debug)

        self.context_radius_factor = context_radius_factor
        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.meters_per_pixel = meters_per_pixel

        self.point_cloud_cache: typing.Dict[str, np.ndarray] = dict()
        self.kdtree_cache: typing.Dict[str, object] = dict()

        # fill GT cache
        if self.load_gt:
            for shape_name in self.shape_names_raw:
                shape_name, start_id, end_id = shape_name.split(',')
                self.create_cache(file_name=shape_name, in_file=self.in_file)

    @staticmethod
    def get_hm_bin_file(dataset_dir: str, file_name: str):
        return os.path.join(dataset_dir, 'bins', file_name, 'heightmaps.bin')

    @staticmethod
    def get_rgb_bin_files(dataset_dir: str, file_name: str):
        return [os.path.join(dataset_dir, 'bins', file_name, 'rgb_{}.bin'.format(i)) for i in range(3)]

    @staticmethod
    def get_hm_query_pts_cache_file(dataset_dir: str, file_name: str):
        return os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_query.npy')

    @staticmethod
    def get_hm_cache_file(dataset_dir: str, file_name: str):
        return os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_hm.npy')

    @staticmethod
    def get_rgb_cache_file(dataset_dir: str, file_name: str):
        return os.path.join(dataset_dir, 'cache_gt', file_name, 'rgb.npy')

    def create_cache(self, file_name: str, in_file: str):
        from source.base.fs import make_dir_for_file, call_necessary
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir

        if not in_file_is_dataset(in_file):
            return  # GT data only for datasets

        dataset_dir = get_dataset_dir(in_file)
        hm_file = self.get_hm_bin_file(dataset_dir, file_name)
        query_pts_file_cache = self.get_hm_query_pts_cache_file(dataset_dir, file_name)
        hm_file_cache = self.get_hm_cache_file(dataset_dir, file_name)
        rgb_map_files = self.get_rgb_bin_files(dataset_dir, file_name)
        rgb_map_files_cache = self.get_rgb_cache_file(dataset_dir, file_name)

        expected_inputs = [hm_file]
        expected_outputs = [query_pts_file_cache, hm_file_cache]
        if os.path.exists(rgb_map_files[0]):
            expected_inputs += rgb_map_files
            expected_outputs += [rgb_map_files_cache]
        if call_necessary(expected_inputs, expected_outputs):
            print('Creating cache for {}'.format(file_name))
            dt = np.dtype('3f8, ({},{})f4'.format(self.hm_size, self.hm_size))
            hm_data = np.fromfile(file=hm_file, dtype=dt)
            query_pts = hm_data['f0']
            make_dir_for_file(query_pts_file_cache)
            np.save(query_pts_file_cache, query_pts)

            hm = hm_data['f1']
            make_dir_for_file(hm_file_cache)
            np.save(hm_file_cache, hm)

            if os.path.exists(rgb_map_files[0]):
                rgb_maps = []
                for rgb_map_file, rgb_map_file_cache in zip(rgb_map_files, rgb_map_files_cache):
                    rgb_map = np.fromfile(file=rgb_map_file, dtype=dt)
                    rgb_map = rgb_map['f1']
                    rgb_maps.append(rgb_map)
                rgb_maps = np.stack(rgb_maps, axis=1)
                rgb_maps /= 255.0
                make_dir_for_file(rgb_map_files_cache)
                np.save(rgb_map_files_cache, rgb_maps)

    @override
    def get_shape_names(self, in_file: str):
        from source.dataloaders.base_data_module import in_file_is_dataset
        super().get_shape_names(in_file)
        self.shape_names_raw = self.shape_names

        if in_file_is_dataset(in_file):
            # duplicate shape names for ids
            shape_names = []
            for shape_name in self.shape_names:
                shape_name, start_id, end_id = shape_name.split(',')
                for i in range(int(start_id), int(end_id), self.dataset_step):
                    shape_names.append(shape_name + ',{},{}'.format(i, i+self.dataset_step))
            self.shape_names = shape_names

    def _get_query_and_hm(self, file_name: str, start_id: int, end_id: int, in_file: str):
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir
        if not in_file_is_dataset(in_file):
            raise NotImplementedError()  # GT data only for datasets

        dataset_dir = get_dataset_dir(in_file)
        query_pts_cache_file = self.get_hm_query_pts_cache_file(dataset_dir, file_name)
        hm_cache_file = self.get_hm_cache_file(dataset_dir, file_name)
        rgb_cache_file = self.get_rgb_cache_file(dataset_dir, file_name)

        def _memmap_to_array(memmap):
            return np.asarray(memmap[start_id:end_id]).copy()

        query_pts = _memmap_to_array(np.load(query_pts_cache_file, mmap_mode='r'))
        hm_arr = _memmap_to_array(np.load(hm_cache_file, mmap_mode='r'))
        if os.path.exists(rgb_cache_file):
            rgb_maps = _memmap_to_array(np.load(rgb_cache_file, mmap_mode='r'))
        else:
            rgb_maps = np.full((end_id - start_id, 3, self.hm_size, self.hm_size), np.nan)

        return query_pts, hm_arr, rgb_maps

    def _get_local_subsamples_fixed_radius_all_pts(
            self, chunk_pts_ms: np.ndarray, chunk_pts_rgb: typing.Optional[np.ndarray],
            pts_query_ms: np.ndarray, shape_id: str):
        from source.base.proximity import query_ball_kdtree

        if not self.load_gt:
            min_point_count = 100  # at least some points in the patch (>4 for triangulation)
        else:
            min_point_count = None

        def _get_from_cache_or_load(requested_file: str, chunk_pts_ms: np.ndarray):
            if requested_file in self.kdtree_cache:
                return self.kdtree_cache[requested_file]
            else:
                from source.base.proximity import make_kdtree
                kdtree = make_kdtree(chunk_pts_ms[:, :2], lib='scipy')
                self.kdtree_cache[requested_file] = kdtree
                return kdtree

        # local patch must be after augmentation
        patch_radius = self._get_patch_radius_p2(hm_res=self.hm_interp_size)
        query_dist_p1 = self._get_patch_radius_p2(hm_res=self.hm_interp_size)
        # kdtree in 2D, take z from local subsample
        dataset_name = self.shape_names[shape_id].split(',')[0]
        kdtree = _get_from_cache_or_load(requested_file=dataset_name, chunk_pts_ms=chunk_pts_ms)
        # we can and should use all points in the radius for interpolation
        patch_pts_ids_lists = query_ball_kdtree(
            kdtree=kdtree, pts_query=pts_query_ms[:, :2], r=query_dist_p1, workers=1, return_sorted=True)
        patch_pts_ids_list = [np.array(ids) for ids in patch_pts_ids_lists]
        if min_point_count is not None:
            valid_num_sub_samples = [len(ids) > min_point_count for ids in patch_pts_ids_list]
            pts_local_ms = [chunk_pts_ms[ids] if valid_num_sub_samples[i] else np.full((1, 3), np.nan)
                            for i, ids in enumerate(patch_pts_ids_list)]
            if chunk_pts_rgb is not None:
                pts_local_rgb = [chunk_pts_rgb[ids] if valid_num_sub_samples[i] else np.full((1, 3), np.nan)
                                 for i, ids in enumerate(patch_pts_ids_list)]
            else:
                pts_local_rgb = None
        else:
            # dataset is wrong if you see an error here
            if any([p.shape[0] == 0 for p in patch_pts_ids_list]):
                print('Error: no points in patch')
            pts_local_ms = [chunk_pts_ms[ids] for ids in patch_pts_ids_list]
            pts_local_rgb = [chunk_pts_rgb[ids] for ids in patch_pts_ids_list]

        # replace query z with mean z of local points, making the heightmap relative to the patch center
        # this way, we're independent of arbitrary query z
        pts_local_ms_z_mean = np.array([np.mean(pts[..., 2]) for pts in pts_local_ms])
        # TODO: find out why this breaks things. maybe overfitting, too small dataset?
        # pts_query_ms[:, 2] = pts_local_ms_z_mean

        subsample_data = (pts_local_ms, pts_local_rgb, pts_query_ms, pts_local_ms_z_mean, patch_radius)
        return subsample_data

    def _make_local_sub_samples(self, shape_id, shape_data: dict) -> dict:
        from source.base.normalization import model_space_to_patch_space_list

        chunk_pts_ms = shape_data['pts_ms']
        chunk_pts_rgb = shape_data['pts_rgb']
        pts_query_ms = shape_data['pts_query_ms']
        numerical_stability_factor = shape_data['numerical_stability_factor']

        # get random subsample
        # subsample_factor = 16
        # chunk_pts_ms = chunk_pts_ms[::subsample_factor]

        local_data = self._get_local_subsamples_fixed_radius_all_pts(
            pts_query_ms=pts_query_ms, chunk_pts_ms=chunk_pts_ms, chunk_pts_rgb=chunk_pts_rgb, shape_id=shape_id)
        (pts_local_ms, pts_local_rgb, pts_query_ms, pts_local_ms_z_mean, patch_radius_interp_ms) = local_data

        # remove query points that find no local subsample
        useless_query_pts = np.array([np.isnan(pts).any() for pts in pts_local_ms])
        pts_query_ms = pts_query_ms[~useless_query_pts]
        pts_local_ms = [pts for pts, useless in zip(pts_local_ms, useless_query_pts) if not useless]
        pts_local_rgb = [pts for pts, useless in zip(pts_local_rgb, useless_query_pts) if not useless]
        pts_local_ms_z_mean = pts_local_ms_z_mean[~useless_query_pts]
        if 'pts_query_ids_xy' in shape_data:
            shape_data['pts_query_ids_xy'] = shape_data['pts_query_ids_xy'][~useless_query_pts]

        patch_radius_hm_ms = self._get_patch_radius_p2(hm_res=self.hm_size)
        pts_local_ps = model_space_to_patch_space_list(
            pts_to_convert_ms=pts_local_ms, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_interp_ms,
            numerical_stability_z_factor=numerical_stability_factor)

        shape_data['pts_query_ms'] = pts_query_ms
        shape_data['patch_radius_interp_ms'] = patch_radius_interp_ms
        shape_data['patch_radius_hm_ms'] = patch_radius_hm_ms
        shape_data['pts_local_ms'] = pts_local_ms
        shape_data['pts_local_ps'] = pts_local_ps
        shape_data['pts_local_rgb'] = pts_local_rgb
        shape_data['pts_local_ms_z_mean'] = pts_local_ms_z_mean
        return shape_data

    def _get_patch_radius_p1(self, hm_res: int):
        hm_length = self.meters_per_pixel * hm_res
        padded_half_edge_length = hm_length * 0.5
        manhattan_dist = 2.0 * padded_half_edge_length
        return manhattan_dist

    def _get_patch_radius_p2(self, hm_res: int):
        hm_diagonal = np.sqrt(2.0) * self.meters_per_pixel * hm_res
        patch_radius = hm_diagonal * 0.5 * self.context_radius_factor
        return patch_radius

    def _load_gt_data(self, shape_id, shape_data):
        shape_name = self.shape_names[shape_id]
        file_name, start_id, end_id = shape_name.split(',')
        start_id = int(start_id)
        end_id = int(end_id)

        # process query pts and heightmap
        pts_query_ms, hm_gt_ms, rgb_maps = self._get_query_and_hm(
            file_name=file_name, start_id=start_id, end_id=end_id, in_file=self.in_file)
        patch_radius_hm_ms = self._get_patch_radius_p2(hm_res=self.hm_size)
        numerical_stability_factor = shape_data['numerical_stability_factor']
        hm_gt_ps = hm_model_space_to_patch_space(
            hm_to_convert_ms=hm_gt_ms, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_hm_ms,
            numerical_stability_z_factor=numerical_stability_factor)

        shape_data['pts_query_ms'] = pts_query_ms
        shape_data['hm_gt_ms'] = hm_gt_ms
        shape_data['hm_gt_ps'] = hm_gt_ps
        shape_data['rgb_gt'] = rgb_maps
        return shape_data

    def _make_rec_data(self, shape_data):

        chunk_pts_ms = shape_data['pts_ms']
        hm_size = self.meters_per_pixel * self.hm_size

        def _sample_query_pts():
            # make xy grid in bounding box of chunk points
            scan_bb = np.array([np.nanmin(chunk_pts_ms, axis=0), np.nanmax(chunk_pts_ms, axis=0)])
            range_x = np.arange(scan_bb[0, 0], scan_bb[1, 0], step=hm_size)
            range_y = np.arange(scan_bb[0, 1], scan_bb[1, 1], step=hm_size)
            pts_query_coords_xy = np.meshgrid(range_x, range_y, indexing='xy')
            pts_query_ms = np.stack([
                pts_query_coords_xy[0], pts_query_coords_xy[1], np.zeros_like(pts_query_coords_xy[0])], axis=-1)
            pts_query_ms = pts_query_ms.reshape(-1, 3)  # from grid of coords to list of coords

            # make ids to grid cells
            pts_query_ids = np.meshgrid(range(range_x.shape[0]), range(range_y.shape[0]))
            pts_query_ids_x = pts_query_ids[0]
            pts_query_ids_y = pts_query_ids[1]
            pts_query_ids_xy = np.stack([pts_query_ids_x.flatten(), pts_query_ids_y.flatten()], axis=-1)
            return pts_query_ms, pts_query_ids_xy

        pts_query_ms, pts_query_ids_xy = _sample_query_pts()
        shape_data['pts_query_ms'] = pts_query_ms
        shape_data['pts_query_ids_xy'] = pts_query_ids_xy
        return shape_data

    @override
    def add_shape_data(self, shape_id, shape_data: dict) -> dict:
        chunk_pts_ms, chunk_pts_rgb = self._read_point_cloud(self.in_file, self.shape_names[shape_id])
        shape_data['pts_ms'] = chunk_pts_ms
        shape_data['pts_rgb'] = chunk_pts_rgb
        shape_data['pc_file_in'] = self.shape_names[shape_id]
        shape_data['meters_per_pixel'] = self.meters_per_pixel

        # const factor to z for numerical stability
        shape_data['numerical_stability_factor'] = 10.0
        return shape_data

    @override
    def add_gt_data(self, shape_id, shape_data: dict) -> dict:
        if self.load_gt:
            shape_data = self._load_gt_data(shape_id, shape_data)
        else:
            shape_data = self._make_rec_data(shape_data)

        shape_data = self._make_local_sub_samples(shape_id, shape_data)
        return shape_data

    @staticmethod
    def aug_keys_if_available(keys, data_keys):
        return [k + '_aug' if k + '_aug' in data_keys else k for k in keys]

    def augment_flip(self, shape_data: dict, keys_to_negate, keys_to_negate_list, keys_to_flip) -> dict:
        batch_size = shape_data[keys_to_negate[0]].shape[0]
        flip_x = self.rng.random_integers(low=0, high=1, size=batch_size).astype(np.bool_)
        flip_y = self.rng.random_integers(low=0, high=1, size=batch_size).astype(np.bool_)

        keys_to_negate = self.aug_keys_if_available(keys_to_negate, shape_data.keys())
        keys_to_negate_list = self.aug_keys_if_available(keys_to_negate_list, shape_data.keys())
        keys_to_flip = self.aug_keys_if_available(keys_to_flip, shape_data.keys())

        shape_data_new = dict()
        for key in keys_to_negate + keys_to_negate_list + keys_to_flip:
            shape_data_new[key] = shape_data[key].copy()

        for key in keys_to_negate:
            shape_data_new[key][flip_x, 0] = -(shape_data_new[key][flip_x, 0])
        for key in keys_to_negate_list:
            for p in range(len(shape_data_new[key])):
                if flip_x[p]:
                    shape_data_new[key][p][:, 0] = -(shape_data_new[key][p][:, 0])
        for key in keys_to_flip:
            arr_flipped = np.flip(shape_data_new[key], axis=-1)
            shape_data_new[key][flip_x] = arr_flipped[flip_x]

        for key in keys_to_negate:
            shape_data_new[key][flip_y, 1] = -(shape_data_new[key][flip_y, 1])
        for key in keys_to_negate_list:
            for p in range(len(shape_data_new[key])):
                if flip_y[p]:
                    shape_data_new[key][p][:, 1] = -(shape_data_new[key][p][:, 1])
        for key in keys_to_flip:
            arr_flipped = np.flip(shape_data_new[key], axis=-2)
            shape_data_new[key][flip_y] = arr_flipped[flip_y]

        for key in shape_data_new.keys():
            shape_data[key + ('_aug' if not key.endswith('_aug') else '')] = shape_data_new[key]

        return shape_data

    def augment_rotate(self, shape_data: dict) -> dict:
        rotations = [0, 1, 2, 3]
        rot90s = self.rng.choice(rotations)

        def _rotate_arr(arr):
            angle_rad = np.radians(90.0 * rot90s)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            arr[:, 0] = cos_angle * arr[:, 0] - sin_angle * arr[:, 0]
            arr[:, 1] = cos_angle * arr[:, 1] - sin_angle * arr[:, 1]
            return arr

        if rot90s > 0:
            shape_data['pts_query_ms'] = _rotate_arr(shape_data['pts_query_ms'])
            shape_data['pts_ms'] = _rotate_arr(shape_data['pts_ms'])

            if self.load_gt:
                hm_gt_ms = shape_data['hm_gt_ms']
                hm_gt_ms = np.rot90(hm_gt_ms, k=rot90s, axes=(1, 2)).copy()
                shape_data['hm_gt_ms'] = hm_gt_ms

        return shape_data

    def augment_z_scale(self, shape_data: dict,
                        keys_to_scale_comp, keys_to_scale_comp_list, keys_to_scale_whole,
                        scale_factor=0.2) -> dict:
        batch_size = shape_data[keys_to_scale_comp[0]].shape[0]
        z_scale = self.rng.uniform(low=1.0 - scale_factor, high=1.0 + scale_factor, size=batch_size)

        keys_to_scale_comp = self.aug_keys_if_available(keys_to_scale_comp, shape_data.keys())
        keys_to_scale_comp_list = self.aug_keys_if_available(keys_to_scale_comp_list, shape_data.keys())
        keys_to_scale_whole = self.aug_keys_if_available(keys_to_scale_whole, shape_data.keys())

        shape_data_new = dict()
        for key in keys_to_scale_comp + keys_to_scale_comp_list + keys_to_scale_whole:
            shape_data_new[key] = shape_data[key].copy()

        for key in keys_to_scale_comp:
            shape_data_new[key][:, 2] *= z_scale
        for key in keys_to_scale_comp_list:
            for p in range(len(shape_data_new[key])):
                shape_data_new[key][p][:, 2] *= z_scale[p]
        for key in keys_to_scale_whole:
            for p in range(len(shape_data_new[key])):
                shape_data_new[key][p] *= z_scale[p]

        for key in shape_data_new.keys():
            shape_data[key + ('_aug' if not key.endswith('_aug') else '')] = shape_data_new[key]

        return shape_data

    @override
    def augment_data(self, shape_data: dict) -> dict:
        keys_to_negate = ['pts_query_ms']
        keys_to_negate_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_flip = ['hm_gt_ms', 'hm_gt_ps'] if self.load_gt else []
        shape_data = self.augment_flip(shape_data, keys_to_negate, keys_to_negate_list, keys_to_flip)

        # shape_data = self.augment_rotate(shape_data)  # broken?

        keys_to_scale_comp = ['pts_query_ms']
        keys_to_scale_comp_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_scale_whole = ['hm_gt_ms', 'hm_gt_ps'] if self.load_gt else []
        shape_data = self.augment_z_scale(shape_data, keys_to_scale_comp, keys_to_scale_comp_list, keys_to_scale_whole)
        return shape_data

    @override
    def visualize_shape_data(self, shape_id, shape_data: dict):
        return

        chunk_pts_ms = shape_data['pts_ms']
        pts_query_ms = shape_data['pts_query_ms']
        pts_local_ms = shape_data['pts_local_ms']
        pts_local_ps = shape_data['pts_local_ps']
        patch_radius_hm_ms = shape_data['patch_radius_hm_ms']
        numerical_stability_factor = shape_data['numerical_stability_factor']

        if not self.debug:
            return

        from source.base.fs import make_dir_for_file
        from source.base.point_cloud import write_ply

        query_pts_file = os.path.join('debug', 'g_pts_query_ms.ply')
        make_dir_for_file(query_pts_file)
        write_ply(file_path=query_pts_file, points=pts_query_ms)
        pts_ms_file = os.path.join('debug', 'g_chunk_pts.ply')
        write_ply(file_path=pts_ms_file, points=chunk_pts_ms)

        if self.load_gt:
            hm_gt_ps = shape_data['hm_gt_ps']
            hm_gt_ms = hm_patch_space_to_model_space(
                hm_to_convert_ps=hm_gt_ps, pts_patch_center_ms=pts_query_ms,
                patch_radius_ms=patch_radius_hm_ms, numerical_stability_z_factor=numerical_stability_factor)
            pts_hm_ms, pts_hm_norm = hm_to_pts(hm_gt_ms, pts_query_ms, pixel_size=10.0)
            hm_ms_file = os.path.join('debug', 'p{}_{}.ply'.format(shape_id, 'hm_gt_ms'))
            write_ply(file_path=hm_ms_file, points=pts_hm_ms, normals=pts_hm_norm)
            pts_hm_ps, pts_hm_norm = hm_to_pts(
                hm_gt_ps, np.zeros_like(pts_query_ms), pixel_size=2.0 / hm_gt_ps.shape[0])
            hm_ps_file = os.path.join('debug', 'p{}_{}.ply'.format(shape_id, 'hm_gt_ps'))
            write_ply(file_path=hm_ps_file, points=pts_hm_ps, normals=pts_hm_norm)

        pts_ms_file = os.path.join('debug', 'p{}_{}.ply'.format(shape_id, 'pts_ms'))
        write_ply(file_path=pts_ms_file, points=pts_local_ms)
        pts_ps_file = os.path.join('debug', 'p{}_{}.ply'.format(shape_id, 'pts_ps'))
        pts_local_ps_no_num_stab = pts_local_ps.copy()
        pts_local_ps_no_num_stab[..., 2] /= numerical_stability_factor
        write_ply(file_path=pts_ps_file, points=pts_local_ps_no_num_stab)

        return

    @override
    def prepare_shape_data_for_cuda(self, shape_data: dict) -> dict:
        shape_data = super(IpesDataset, self).prepare_shape_data_for_cuda(shape_data)
        shape_data.pop('pts_ms')  # raw point cloud is large
        shape_data.pop('pts_rgb')  # raw point cloud is large
        # shape_data.pop('pts_local_ms')  # would be collate of variable length
        # shape_data.pop('pts_local_ps')  # would be collate of variable length
        # shape_data.pop('pts_local_rgb')  # would be collate of variable length
        return shape_data

    def _read_point_cloud(self, in_file: str, pc_file_in: str):
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir

        def _get_from_cache_or_load(requested_file: str):
            if requested_file in self.point_cloud_cache.keys():
                return self.point_cloud_cache[requested_file]
            else:
                chunk_pts = np.loadtxt(requested_file, dtype=np.float64, delimiter=',')
                chunk_pts.flags.writeable = False  # don't mess with the cache
                self.point_cloud_cache[requested_file] = chunk_pts
                return chunk_pts

        pc_file_name = pc_file_in.split(',')[0]
        if in_file_is_dataset(in_file):
            dataset_dir = get_dataset_dir(in_file)
            pts_file = os.path.join(dataset_dir, 'bins', pc_file_name, 'chunkPoints.csv')
            chunk_pts_all = _get_from_cache_or_load(pts_file)
            pts_ply_file = os.path.join(dataset_dir, 'bins', pc_file_name, 'chunkPoints.ply')
        else:
            dataset_dir = os.path.dirname(in_file)
            pts_file = in_file
            chunk_pts_all = _get_from_cache_or_load(pts_file)
            pts_ply_file = os.path.join(dataset_dir, 'chunkPoints.ply')

        has_colors = chunk_pts_all.shape[1] == 6  # assume 0:3 xyz, 3:6 rgb
        chunk_pts_xyz = chunk_pts_all[:, :3]
        chunk_pts_rgb = chunk_pts_all[:, 3:6] if has_colors else np.full(chunk_pts_xyz.shape, np.nan)

        # these have no valid RGB -> TODO: put in higher level parameter
        if in_file_is_dataset(in_file) and pc_file_name in ['swisssurface3d']:
            chunk_pts_rgb = np.full(chunk_pts_xyz.shape, np.nan)
        chunk_pts_rgb = chunk_pts_rgb / 255.0  # normalize to [0, 1]

        # convert to ply for easy visualization in meshlab
        from source.base.fs import call_necessary, make_dir_for_file
        if call_necessary(file_in=pts_file, file_out=pts_ply_file):
            make_dir_for_file(pts_ply_file)
            from source.base.point_cloud import write_ply
            if has_colors:
                write_ply(file_path=pts_ply_file, points=chunk_pts_xyz, colors=chunk_pts_rgb)
            else:
                write_ply(file_path=pts_ply_file, points=chunk_pts_xyz)

        return chunk_pts_xyz, chunk_pts_rgb
