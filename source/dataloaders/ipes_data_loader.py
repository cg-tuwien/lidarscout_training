import os.path
import typing

from overrides import override
import numpy as np

from source.dataloaders.base_data_module import BaseDataModule, BaseDataset
from source.base.math import hm_to_pts
from source.base.normalization import hm_model_space_to_patch_space, hm_patch_space_to_model_space

# shapes whose LAS RGB channel is present but known to be invalid/placeholder data, so it must
# be forced to NaN instead of being used. Shared with source/dataloaders/img_cache_precompute.py.
SHAPES_WITHOUT_VALID_RGB = frozenset({'swisssurface3d'})


def img_cache_key_prefix_for_chunk_pts_file(chunk_pts_file: str, pc_key: str) -> str:
    """
    Cheap (stat-only, no point data read) identifier for a point cloud, used as the shared prefix
    of pts_to_img_cached's cache_key (see that function's docstring for why avoiding a
    content-hash matters). Combining pc_key with chunkPoints' mtime means a regenerated point
    cloud naturally invalidates old cache entries (same shape name, new mtime -> new keys)
    without ever needing to look at the actual point values.
    """
    try:
        mtime = int(os.path.getmtime(chunk_pts_file))
    except OSError:
        mtime = 0
    return '{}|{}'.format(pc_key, mtime)


def img_cache_key_prefix(in_file: str, pc_key: str) -> str:
    """Convenience wrapper for callers that only have in_file/pc_key (not the resolved
    chunkPoints path) -- see img_cache_key_prefix_for_chunk_pts_file."""
    from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir

    if in_file_is_dataset(in_file):
        dataset_dir = get_dataset_dir(in_file)
        chunk_pts_file = os.path.join(dataset_dir, 'bins', pc_key, 'chunkPoints.csv')
    else:
        chunk_pts_file = in_file
    return img_cache_key_prefix_for_chunk_pts_file(chunk_pts_file, pc_key)


class IpesDataModule(BaseDataModule):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 seed, in_file, workers, use_ddp,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, debug: bool, batch_size: int,
                 reconstruction_chunk_size: int = 500):
        super(IpesDataModule, self).__init__(
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size, debug=debug, seed=seed)

        self.context_radius_factor = context_radius_factor
        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.meters_per_pixel = meters_per_pixel
        self.dataset_step = dataset_step
        # chunk size for reconstruction/predict mode's query grid -- see
        # IpesDataset._reconstruction_chunk_names. Irrelevant to fit/test (load_gt=True).
        self.reconstruction_chunk_size = reconstruction_chunk_size

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
            reconstruction_chunk_size=self.reconstruction_chunk_size,
        )
        return dataset


class IpesDataset(BaseDataset):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, seed, use_ddp, load_gt: bool,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug,
                 reconstruction_chunk_size: int = 500):

        self.dataset_step = dataset_step
        self.shape_names_raw = []

        # must be set before super().__init__() -- it calls get_shape_names(), which for
        # reconstruction mode (see _reconstruction_chunk_names) needs these to size the query
        # grid, and needs the caches below to read/cache the point cloud while doing so
        self.context_radius_factor = context_radius_factor
        self.hm_interp_size = hm_interp_size
        self.hm_size = hm_size
        self.meters_per_pixel = meters_per_pixel
        self.reconstruction_chunk_size = reconstruction_chunk_size

        self.point_cloud_cache: typing.Dict[str, np.ndarray] = dict()
        self.kdtree_cache: typing.Dict[str, object] = dict()
        self.local_pts_idx_cache: typing.Dict[str, typing.Tuple[np.ndarray, np.ndarray]] = dict()
        self.rec_query_grid_cache: typing.Dict[str, typing.Tuple[np.ndarray, np.ndarray]] = dict()

        super(IpesDataset, self).__init__(
            in_file=in_file, seed=seed, use_ddp=use_ddp,
            patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation,
            load_gt=load_gt, debug=debug)

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

    @staticmethod
    def _npy_cache_is_valid(cache_file: str) -> bool:
        if not os.path.isfile(cache_file):
            return False
        try:
            np.load(cache_file, mmap_mode='r')
            return True
        except Exception:
            return False

    def create_cache(self, file_name: str, in_file: str, force: bool = False):
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
        outputs_valid = all(self._npy_cache_is_valid(output_file) for output_file in expected_outputs)
        if not force and outputs_valid and not call_necessary(expected_inputs, expected_outputs):
            return

        if not outputs_valid:
            print('Recreating invalid cache for {}'.format(file_name))
        else:
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

        if self.load_gt:
            if in_file_is_dataset(in_file):
                # duplicate shape names for ids
                shape_names = []
                for shape_name in self.shape_names:
                    shape_name, start_id, end_id = shape_name.split(',')
                    for i in range(int(start_id), int(end_id), self.dataset_step):
                        shape_names.append(shape_name + ',{},{}'.format(i, i+self.dataset_step))
                self.shape_names = shape_names
        else:
            # Reconstruction: expand each shape into chunks of its query grid (see
            # _reconstruction_chunk_names) instead of one entry covering the whole shape/point
            # cloud -- lets predict_step (source/modules/ipes_base.py) generate and consume each
            # chunk's local point subsamples lazily, one DataLoader item at a time, instead of
            # materializing every window's resolved local points for the whole reconstruction
            # area at once (which crashed at ~19GB+ on a 100x-densified stress-test shape even
            # after the underlying KDTree query itself was batched).
            shape_names = []
            if in_file_is_dataset(in_file):
                for shape_name in self.shape_names:
                    pc_key = shape_name.split(',')[0]
                    shape_names.extend(self._reconstruction_chunk_names(pc_key))
            else:
                shape_names.extend(self._reconstruction_chunk_names(in_file))
            self.shape_names = shape_names

    def _sample_rec_query_pts(self, chunk_pts_ms: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Builds the full reconstruction query-point grid (coordinates only, cheap) covering the
        bounding box of `chunk_pts_ms`. Factored out of the old _make_rec_data so it can be
        computed once per point cloud and cached (see _get_or_build_rec_query_grid), instead of
        being recomputed -- and, more importantly, instead of every one of its windows' local
        point subsamples being resolved eagerly -- for every reconstruction chunk."""
        hm_size = self.meters_per_pixel * self.hm_size
        scan_bb = np.array([np.nanmin(chunk_pts_ms, axis=0), np.nanmax(chunk_pts_ms, axis=0)])
        range_x = np.arange(scan_bb[0, 0], scan_bb[1, 0], step=hm_size)
        range_y = np.arange(scan_bb[0, 1], scan_bb[1, 1], step=hm_size)
        pts_query_coords_xy = np.meshgrid(range_x, range_y, indexing='xy')
        pts_query_ms = np.stack([
            pts_query_coords_xy[0], pts_query_coords_xy[1], np.zeros_like(pts_query_coords_xy[0])], axis=-1)
        pts_query_ms = pts_query_ms.reshape(-1, 3)  # from grid of coords to list of coords

        pts_query_ids = np.meshgrid(range(range_x.shape[0]), range(range_y.shape[0]))
        pts_query_ids_x = pts_query_ids[0]
        pts_query_ids_y = pts_query_ids[1]
        pts_query_ids_xy = np.stack([pts_query_ids_x.flatten(), pts_query_ids_y.flatten()], axis=-1)
        return pts_query_ms, pts_query_ids_xy

    def _get_or_build_rec_query_grid(self, pc_key: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Cached per point cloud (pc_key) so the (cheap) grid computation and the (not-so-cheap,
        but itself cached via point_cloud_cache/load_csv_points_cached) point-cloud read only
        happen once per shape, no matter how many chunks it's split into.

        Also eagerly builds and caches the KDTree for this point cloud here, in self.kdtree_cache
        -- not lazily on first use inside a DataLoader worker, which is where it used to happen.
        This method runs from get_shape_names(), called from __init__, i.e. in the main process
        *before* the DataLoader forks its worker processes. On Linux/WSL (fork is the default
        multiprocessing start method, unlike Windows' spawn-only), each forked worker inherits
        the parent's memory via copy-on-write, so a KDTree built here is shared read-only across
        every worker for free -- no per-worker rebuild, no extra memory, since queries never
        mutate it (the same property that already makes workers=-1 query parallelism safe).
        Building it lazily per-worker (the old behavior, still what happens on Windows since
        spawn workers don't inherit parent memory at all) meant N workers independently paid the
        full build cost for the same 35M-point tree -- confirmed empirically to nearly OOM a
        36-core WSL run even at just 8 workers."""
        if pc_key not in self.rec_query_grid_cache:
            chunk_pts_ms, _ = self._read_point_cloud(self.in_file, pc_key)
            self.rec_query_grid_cache[pc_key] = self._sample_rec_query_pts(chunk_pts_ms)
            if pc_key not in self.kdtree_cache:
                from source.base.proximity import make_kdtree
                self.kdtree_cache[pc_key] = make_kdtree(chunk_pts_ms[:, :2], lib='scipy')
        return self.rec_query_grid_cache[pc_key]

    def _reconstruction_chunk_names(self, pc_key: str) -> typing.List[str]:
        pts_query_ms, _ = self._get_or_build_rec_query_grid(pc_key)
        num_query_pts = pts_query_ms.shape[0]
        chunk = max(1, int(self.reconstruction_chunk_size))
        return [
            '{},{},{}'.format(pc_key, start, min(start + chunk, num_query_pts))
            for start in range(0, num_query_pts, chunk)
        ]

    def _get_query_and_hm(self, file_name: str, start_id: int, end_id: int, in_file: str):
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir
        if not in_file_is_dataset(in_file):
            raise NotImplementedError()  # GT data only for datasets

        dataset_dir = get_dataset_dir(in_file)
        query_pts_cache_file = self.get_hm_query_pts_cache_file(dataset_dir, file_name)
        hm_cache_file = self.get_hm_cache_file(dataset_dir, file_name)
        rgb_cache_file = self.get_rgb_cache_file(dataset_dir, file_name)

        if not self._npy_cache_is_valid(query_pts_cache_file) or not self._npy_cache_is_valid(hm_cache_file) or (
            os.path.exists(rgb_cache_file) and not self._npy_cache_is_valid(rgb_cache_file)
        ):
            self.create_cache(file_name=file_name, in_file=in_file, force=True)

        def _memmap_to_array(memmap):
            return np.asarray(memmap[start_id:end_id]).copy()

        query_pts = _memmap_to_array(np.load(query_pts_cache_file, mmap_mode='r'))
        hm_arr = _memmap_to_array(np.load(hm_cache_file, mmap_mode='r'))
        if os.path.exists(rgb_cache_file):
            rgb_maps = _memmap_to_array(np.load(rgb_cache_file, mmap_mode='r'))
        else:
            rgb_maps = np.full((end_id - start_id, 3, self.hm_size, self.hm_size), np.nan)

        return query_pts, hm_arr, rgb_maps

    def _get_local_subsamples_fixed_radius_indexed(
            self, chunk_pts_ms: np.ndarray, shape_id: str) -> typing.Optional[typing.List[np.ndarray]]:
        """
        Fast path for GT-loading (fit/test) only: looks up this window's local point ids from a
        per-shape cache persisted to disk once (see source/dataloaders/local_points_cache.py)
        instead of building a KDTree in this worker process. Returns None if the shape isn't a
        dataset entry (e.g. a raw point cloud path used standalone) -- caller falls back to the
        on-the-fly path.

        Deliberately not extended to reconstruction (predict): a point cloud is only ever
        reconstructed once, so there's nothing to amortize a disk-persisted cache against -- it
        would only add write cost and on-disk footprint for no reuse. Reconstruction still avoids
        rebuilding the KDTree across chunks of the *same* run via self.kdtree_cache below (an
        in-memory, this-process-lifetime cache, not a disk one).
        """
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir
        from source.dataloaders.local_points_cache import (
            get_patch_radius, load_or_build_local_point_index_cache, slice_local_point_ids)

        if not self.load_gt or not in_file_is_dataset(self.in_file):
            return None

        dataset_name, start_id, end_id = self.shape_names[shape_id].split(',')
        start_id, end_id = int(start_id), int(end_id)
        dataset_dir = get_dataset_dir(self.in_file)
        patch_radius = get_patch_radius(self.hm_interp_size, self.context_radius_factor, self.meters_per_pixel)

        if dataset_name not in self.local_pts_idx_cache:
            chunk_pts_file = os.path.join(dataset_dir, 'bins', dataset_name, 'chunkPoints.csv')
            self.local_pts_idx_cache[dataset_name] = load_or_build_local_point_index_cache(
                dataset_dir=dataset_dir, shape_name=dataset_name, chunk_pts_source_file=chunk_pts_file,
                # only actually loaded if the cache is missing/stale -- see docstring above
                get_chunk_pts_xy=lambda: chunk_pts_ms[:, :2],
                get_query_pts_xy=lambda: np.load(
                    os.path.join(dataset_dir, 'cache_gt', dataset_name, 'heightmaps_query.npy'))[:, :2],
                patch_radius=patch_radius)
        indices, offsets = self.local_pts_idx_cache[dataset_name]
        return slice_local_point_ids(indices, offsets, start_id, end_id)

    def _get_local_subsamples_fixed_radius_all_pts(
            self, chunk_pts_ms: np.ndarray, chunk_pts_rgb: typing.Optional[np.ndarray],
            pts_query_ms: np.ndarray, shape_id: str):
        from source.base.proximity import query_ball_kdtree_batched

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

        patch_pts_ids_list = None
        if self.load_gt:
            # avoids building a KDTree in this worker at all when the shape's cache is warm --
            # see _get_local_subsamples_fixed_radius_indexed's docstring for why this matters
            patch_pts_ids_list = self._get_local_subsamples_fixed_radius_indexed(
                chunk_pts_ms=chunk_pts_ms, shape_id=shape_id)

        if patch_pts_ids_list is None:
            # kdtree in 2D, take z from local subsample
            dataset_name = self.shape_names[shape_id].split(',')[0]
            kdtree = _get_from_cache_or_load(requested_file=dataset_name, chunk_pts_ms=chunk_pts_ms)
            # we can and should use all points in the radius for interpolation
            # batched: an all-at-once query_ball_point() call for every query point (e.g.
            # predict/reconstruction mode's on-the-fly _make_rec_data grid -- a point cloud is
            # only ever reconstructed once, so this is deliberately not routed through
            # local_points_cache.py's disk-persisted cache; there's no reuse to amortize a
            # write against) is itself a memory bottleneck inside scipy -- see
            # query_ball_kdtree_batched's docstring, this crashed the system at ~18GB+ on a dense
            # reconstruction shape. Reconstruction only ever asks for one small chunk's worth of
            # query points at a time (see _reconstruction_chunk_names), so query volume is
            # bounded regardless of the workers setting below.
            #
            # workers=1 (not -1/all-cores) unconditionally, for both GT and reconstruction: the
            # real parallelism is meant to come from the DataLoader's own worker processes (each
            # chunk/window-group is independent, so multiple can render concurrently across
            # worker processes -- reconstruction's chunking made this viable, see
            # IpesDataset._reconstruction_chunk_names). If this per-call query also used
            # workers=-1, every one of those worker *processes* would additionally try to use
            # every core for its own query, oversubscribing CPUs badly (N worker processes each
            # spawning up to core_count threads). One core per DataLoader worker, N DataLoader
            # workers, is the safe way to actually use N cores.
            patch_pts_ids_list = query_ball_kdtree_batched(
                kdtree=kdtree, pts_query=pts_query_ms[:, :2], r=query_dist_p1, batch_size=500,
                workers=1, return_sorted=True)
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
        sun_pos_xy = shape_data['sun_pos_xy']
        pts_query_ms = shape_data['pts_query_ms']
        numerical_stability_factor = shape_data['numerical_stability_factor']

        # Expand per-shape sun direction to one vector per query patch.
        sun_pos_xy = np.broadcast_to(sun_pos_xy[np.newaxis], (pts_query_ms.shape[0], 2)).copy()

        # get random subsample
        # subsample_factor = 16
        # chunk_pts_ms = chunk_pts_ms[::subsample_factor]

        local_data = self._get_local_subsamples_fixed_radius_all_pts(
            pts_query_ms=pts_query_ms, chunk_pts_ms=chunk_pts_ms, chunk_pts_rgb=chunk_pts_rgb, shape_id=shape_id)
        (pts_local_ms, pts_local_rgb, pts_query_ms, pts_local_ms_z_mean, patch_radius_interp_ms) = local_data

        # remove query points that find no local subsample
        useless_query_pts = np.array([np.isnan(pts).any() for pts in pts_local_ms])
        pts_query_ms = pts_query_ms[~useless_query_pts]
        sun_pos_xy = sun_pos_xy[~useless_query_pts]
        pts_local_ms = [pts for pts, useless in zip(pts_local_ms, useless_query_pts) if not useless]
        pts_local_rgb = [pts for pts, useless in zip(pts_local_rgb, useless_query_pts) if not useless]
        pts_local_ms_z_mean = pts_local_ms_z_mean[~useless_query_pts]
        if 'pts_query_ids_xy' in shape_data:
            shape_data['pts_query_ids_xy'] = shape_data['pts_query_ids_xy'][~useless_query_pts]
        if 'query_abs_ids' in shape_data:
            shape_data['query_abs_ids'] = shape_data['query_abs_ids'][~useless_query_pts]

        patch_radius_hm_ms = self._get_patch_radius_p2(hm_res=self.hm_size)
        pts_local_ps = model_space_to_patch_space_list(
            pts_to_convert_ms=pts_local_ms, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_interp_ms,
            numerical_stability_z_factor=numerical_stability_factor)

        shape_data['pts_query_ms'] = pts_query_ms
        shape_data['sun_pos_xy'] = sun_pos_xy
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
        # cheap identifiers for pts_to_img_cached's fast cache-key path (see img_cache_key_prefix)
        shape_data['img_cache_key_prefix'] = img_cache_key_prefix(self.in_file, file_name)
        shape_data['query_abs_ids'] = np.arange(start_id, end_id)
        return shape_data

    def _make_rec_data(self, shape_data: dict, pc_key: str, start_id: int, end_id: int) -> dict:
        # only slices the (cached, cheap -- coordinates only) full query grid for this point
        # cloud; resolving the local point subsamples for just this chunk happens afterwards in
        # _make_local_sub_samples, called once per chunk from add_gt_data below
        pts_query_ms_full, pts_query_ids_xy_full = self._get_or_build_rec_query_grid(pc_key)
        shape_data['pts_query_ms'] = pts_query_ms_full[start_id:end_id]
        shape_data['pts_query_ids_xy'] = pts_query_ids_xy_full[start_id:end_id]
        # cheap identifiers for pts_to_img_cached's fast cache-key path (see img_cache_key_prefix)
        shape_data['img_cache_key_prefix'] = img_cache_key_prefix(self.in_file, pc_key)
        shape_data['query_abs_ids'] = np.arange(start_id, end_id)
        # tells predict_step (source/modules/ipes_base.py) when to reset/flush its cross-chunk
        # result accumulator -- plain python scalars, converted to 0-dim tensors downstream by
        # dict_np_to_torch same as numerical_stability_factor above
        shape_data['rec_chunk_start_id'] = int(start_id)
        shape_data['rec_is_last_chunk'] = bool(end_id >= pts_query_ms_full.shape[0])
        return shape_data

    @override
    def add_shape_data(self, shape_id, shape_data: dict) -> dict:
        shape_name_full = self.shape_names[shape_id]
        chunk_pts_ms, chunk_pts_rgb = self._read_point_cloud(self.in_file, shape_name_full)
        shape_data['pts_ms'] = chunk_pts_ms
        shape_data['pts_rgb'] = chunk_pts_rgb
        shape_data['sun_pos_xy'] = np.array([0.0, -1.0], dtype=np.float32)
        # reconstruction shape names carry a ',start,end' chunk suffix (see
        # _reconstruction_chunk_names) -- strip it so predict_step's output-file naming stays
        # stable across all chunks of the same point cloud, instead of one output file per chunk
        shape_data['pc_file_in'] = shape_name_full.split(',')[0] if not self.load_gt else shape_name_full
        shape_data['meters_per_pixel'] = self.meters_per_pixel

        # const factor to z for numerical stability
        shape_data['numerical_stability_factor'] = 10.0
        return shape_data

    @override
    def add_gt_data(self, shape_id, shape_data: dict) -> dict:
        if self.load_gt:
            shape_data = self._load_gt_data(shape_id, shape_data)
        else:
            pc_key, start_id, end_id = self.shape_names[shape_id].split(',')
            shape_data = self._make_rec_data(shape_data, pc_key=pc_key, start_id=int(start_id), end_id=int(end_id))

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

    def augment_rotate(self, shape_data: dict, keys_to_rotate_grid, keys_to_rotate_pts) -> dict:
        keys_to_rotate_grid = self.aug_keys_if_available(keys_to_rotate_grid, shape_data.keys())
        keys_to_rotate_pts = self.aug_keys_if_available(keys_to_rotate_pts, shape_data.keys())

        if len(keys_to_rotate_grid) + len(keys_to_rotate_pts) == 0:
            return shape_data

        batch_key = (keys_to_rotate_grid + keys_to_rotate_pts)[0]
        batch_data = shape_data[batch_key]
        batch_size = len(batch_data) if isinstance(batch_data, list) else batch_data.shape[0]

        rotations = [0, 1, 2, 3]
        rot90s = self.rng.choice(rotations, size=batch_size)

        def _rotate_pts_array(arr: np.ndarray, k: int) -> np.ndarray:
            if k == 0:
                return arr
            x = arr[..., 0].copy()
            y = arr[..., 1].copy()
            if k == 1:
                arr[..., 0] = -y
                arr[..., 1] = x
            elif k == 2:
                arr[..., 0] = -x
                arr[..., 1] = -y
            else:
                arr[..., 0] = y
                arr[..., 1] = -x
            return arr

        def _rotate_grid_batch(arr: np.ndarray) -> np.ndarray:
            arr_rot = arr.copy()
            for i, k in enumerate(rot90s):
                if k != 0:
                    # Rotate only spatial dims H/W for each item in batch.
                    arr_rot[i] = np.rot90(arr_rot[i], k=int(k), axes=(-2, -1))
            return arr_rot

        def _rotate_pts_batch(arr: np.ndarray) -> np.ndarray:
            arr_rot = arr.copy()
            if arr_rot.shape[0] != batch_size:
                raise ValueError('Point batch has mismatched batch dimension for rotation.')
            for i, k in enumerate(rot90s):
                if k != 0:
                    arr_rot[i] = _rotate_pts_array(arr_rot[i], int(k))
            return arr_rot

        def _rotate_pts_list(arr_list: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
            if len(arr_list) != batch_size:
                raise ValueError('Point list has mismatched batch length for rotation.')
            arr_list_rot = [arr.copy() for arr in arr_list]
            for i, k in enumerate(rot90s):
                if k != 0:
                    arr_list_rot[i] = _rotate_pts_array(arr_list_rot[i], int(k))
            return arr_list_rot

        shape_data_new = dict()
        for key in keys_to_rotate_grid + keys_to_rotate_pts:
            shape_data_new[key] = shape_data[key].copy()

        for key in keys_to_rotate_grid:
            shape_data_new[key] = _rotate_grid_batch(shape_data_new[key])

        for key in keys_to_rotate_pts:
            if isinstance(shape_data_new[key], list):
                shape_data_new[key] = _rotate_pts_list(shape_data_new[key])
            else:
                shape_data_new[key] = _rotate_pts_batch(shape_data_new[key])

        for key in shape_data_new.keys():
            shape_data[key + ('_aug' if not key.endswith('_aug') else '')] = shape_data_new[key]

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
        keys_to_negate = ['pts_query_ms', 'sun_pos_xy']
        keys_to_negate_list = ['pts_local_ms', 'pts_local_ps']
        keys_to_flip = ['hm_gt_ms', 'hm_gt_ps'] if self.load_gt else []
        shape_data = self.augment_flip(shape_data, keys_to_negate, keys_to_negate_list, keys_to_flip)

        # shape_data = self.augment_rotate(shape_data)  # broken?

        # distorts understanding of heights
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
        # default=None: IpesImgDataset's cache-only fast path (see _check_fully_cached) never
        # sets these in the first place -- it skips reading the point cloud entirely when every
        # image the item needs is already cached, so there's nothing here to pop in that case
        shape_data.pop('pts_ms', None)  # raw point cloud is large
        shape_data.pop('pts_rgb', None)  # raw point cloud is large
        # shape_data.pop('pts_local_ms')  # would be collate of variable length
        # shape_data.pop('pts_local_ps')  # would be collate of variable length
        # shape_data.pop('pts_local_rgb')  # would be collate of variable length
        return shape_data

    def _read_point_cloud(self, in_file: str, pc_file_in: str):
        from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir

        def _get_from_cache_or_load(requested_file: str):
            from source.base.fs import load_csv_points_cached

            if requested_file in self.point_cloud_cache.keys():
                return self.point_cloud_cache[requested_file]
            else:
                # load_csv_points_cached also maintains its own on-disk .npy cache next to the
                # CSV, so repeat reads (including from other processes/DataLoader workers) skip
                # text parsing entirely -- the in-memory self.point_cloud_cache above only helps
                # within this one worker process.
                # mmap_mode='r': on a cache hit, memory-map instead of fully loading -- separate
                # worker processes mmap-ing the same file share physical pages via the OS page
                # cache, so this is what actually avoids N-way RAM duplication of a shape's raw
                # points across workers (see source/dataloaders/local_points_cache.py's docstring
                # for the fuller story -- this pairs with that cache to remove both the KDTree
                # and the raw-array duplication that caused workers>0 to exhaust memory).
                chunk_pts = load_csv_points_cached(requested_file, mmap_mode='r')
                if chunk_pts.flags.writeable:
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

        # these have no valid RGB (placeholder/garbage data despite having a color column)
        if in_file_is_dataset(in_file) and pc_file_name in SHAPES_WITHOUT_VALID_RGB:
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
