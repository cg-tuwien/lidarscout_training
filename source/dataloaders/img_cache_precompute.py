import os
import shutil
import typing
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from source.base.normalization import model_space_to_patch_space_list
from source.base.point_cloud import pts_to_img_cached
from source.dataloaders.ipes_data_loader import SHAPES_WITHOUT_VALID_RGB


def _read_shape_list(shape_list_file: str) -> typing.List[str]:
    if not os.path.exists(shape_list_file):
        return []
    with open(shape_list_file) as f:
        shape_names = [x.strip() for x in f.readlines()]
    return [x for x in shape_names if x]


def _expand_shape_ranges(shape_names_raw: typing.List[str], dataset_step: int) -> typing.List[typing.Tuple[str, int, int]]:
    expanded: typing.List[typing.Tuple[str, int, int]] = []
    for shape_name in shape_names_raw:
        file_name, start_id, end_id = shape_name.split(',')
        start = int(start_id)
        end = int(end_id)
        for i in range(start, end, dataset_step):
            expanded.append((file_name, i, i + dataset_step))
    return expanded


def _ensure_gt_cache_for_shape(dataset_dir: str, file_name: str, hm_size: int):
    from source.base.fs import make_dir_for_file

    hm_file = os.path.join(dataset_dir, 'bins', file_name, 'heightmaps.bin')
    query_pts_file_cache = os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_query.npy')
    hm_file_cache = os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_hm.npy')
    rgb_map_files = [os.path.join(dataset_dir, 'bins', file_name, f'rgb_{i}.bin') for i in range(3)]
    rgb_map_file_cache = os.path.join(dataset_dir, 'cache_gt', file_name, 'rgb.npy')

    def _is_valid_npy(cache_file: str) -> bool:
        if not os.path.isfile(cache_file):
            return False
        try:
            np.load(cache_file, mmap_mode='r')
            return True
        except Exception:
            return False

    if _is_valid_npy(query_pts_file_cache) and _is_valid_npy(hm_file_cache) and (
        not os.path.exists(rgb_map_file_cache) or _is_valid_npy(rgb_map_file_cache)
    ):
        return

    dt = np.dtype(f'3f8, ({hm_size},{hm_size})f4')
    hm_data = np.fromfile(file=hm_file, dtype=dt)
    query_pts = hm_data['f0']
    hm = hm_data['f1']
    make_dir_for_file(query_pts_file_cache)
    np.save(query_pts_file_cache, query_pts)
    make_dir_for_file(hm_file_cache)
    np.save(hm_file_cache, hm)

    if os.path.exists(rgb_map_files[0]) and not os.path.exists(rgb_map_file_cache):
        rgb_maps = []
        for rgb_map_file in rgb_map_files:
            rgb_map = np.fromfile(file=rgb_map_file, dtype=dt)
            rgb_maps.append(rgb_map['f1'])
        rgb_maps = np.stack(rgb_maps, axis=1)
        rgb_maps /= 255.0
        make_dir_for_file(rgb_map_file_cache)
        np.save(rgb_map_file_cache, rgb_maps)


def _chunk_points_and_rgb(dataset_dir: str, file_name: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    from source.base.fs import load_csv_points_cached

    # this is called once per (shape, id-range) precompute task, i.e. many times per shape --
    # the on-disk .npy cache means only the very first task to touch a given shape pays for
    # parsing its chunkPoints.csv as text; every later task (any worker) gets a fast binary load
    # mmap_mode='r': shares physical pages across worker processes via the OS page cache instead
    # of each of this pool's workers fully loading its own copy (see local_points_cache.py)
    pts_file = os.path.join(dataset_dir, 'bins', file_name, 'chunkPoints.csv')
    chunk_pts_all = load_csv_points_cached(pts_file, mmap_mode='r')
    has_colors = chunk_pts_all.shape[1] == 6
    chunk_pts_xyz = chunk_pts_all[:, :3]
    chunk_pts_rgb = chunk_pts_all[:, 3:6] if has_colors else np.full(chunk_pts_xyz.shape, np.nan)
    if file_name in SHAPES_WITHOUT_VALID_RGB:
        chunk_pts_rgb = np.full(chunk_pts_xyz.shape, np.nan)
    chunk_pts_rgb = chunk_pts_rgb / 255.0
    return chunk_pts_xyz, chunk_pts_rgb


def _load_query_pts(dataset_dir: str, file_name: str, start_id: int, end_id: int) -> np.ndarray:
    query_pts_cache_file = os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_query.npy')
    query_pts = np.asarray(np.load(query_pts_cache_file, mmap_mode='r')[start_id:end_id]).copy()
    return query_pts


def _task_worker(task: dict) -> typing.Tuple[str, int]:
    dataset_dir = task['dataset_dir']
    hm_interp_size = task['hm_interp_size']
    context_radius_factor = task['context_radius_factor']
    meters_per_pixel = task['meters_per_pixel']
    pts_to_img_methods = task['pts_to_img_methods']
    rgb_to_img_methods = task['rgb_to_img_methods']
    hm_size = task['hm_size']
    file_name = task['file_name']
    start_id = task['start_id']
    end_id = task['end_id']

    cache_dir = os.path.join(dataset_dir, 'img_cache')

    _ensure_gt_cache_for_shape(dataset_dir=dataset_dir, file_name=file_name, hm_size=hm_size)

    chunk_pts_ms, chunk_pts_rgb = _chunk_points_and_rgb(dataset_dir=dataset_dir, file_name=file_name)
    pts_query_ms = _load_query_pts(dataset_dir=dataset_dir, file_name=file_name, start_id=start_id, end_id=end_id)

    # cached per shape (see local_points_cache.py): avoids every one of this pool's (shape,
    # id-range) tasks independently building its own KDTree over the same shape's points
    from source.dataloaders.local_points_cache import (
        get_patch_radius, load_or_build_local_point_index_cache, slice_local_point_ids)
    patch_radius = get_patch_radius(hm_interp_size, context_radius_factor, meters_per_pixel)
    chunk_pts_file = os.path.join(dataset_dir, 'bins', file_name, 'chunkPoints.csv')
    query_pts_file = os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_query.npy')
    indices, offsets = load_or_build_local_point_index_cache(
        dataset_dir=dataset_dir, shape_name=file_name, chunk_pts_source_file=chunk_pts_file,
        get_chunk_pts_xy=lambda: chunk_pts_ms[:, :2],
        get_query_pts_xy=lambda: np.asarray(np.load(query_pts_file)[:, :2]),
        patch_radius=patch_radius)
    patch_pts_ids_lists = slice_local_point_ids(indices, offsets, start_id, end_id)

    # cheap, stat-only cache-key prefix (see img_cache_key_prefix_for_chunk_pts_file's docstring
    # for why this replaced hashing each patch's actual point content)
    from source.dataloaders.ipes_data_loader import img_cache_key_prefix_for_chunk_pts_file
    cache_key_prefix = img_cache_key_prefix_for_chunk_pts_file(chunk_pts_file, file_name)

    min_point_count = 100
    pts_local_ms = []
    pts_local_rgb = []
    pts_query_ms_valid = []
    query_abs_ids_valid = []
    for i, ids in enumerate(tqdm(patch_pts_ids_lists, desc='Filtering points', leave=False)):
        if len(ids) > min_point_count:
            ids_np = np.asarray(ids)
            pts_local_ms.append(chunk_pts_ms[ids_np])
            pts_local_rgb.append(chunk_pts_rgb[ids_np])
            pts_query_ms_valid.append(pts_query_ms[i])
            query_abs_ids_valid.append(start_id + i)

    if len(pts_local_ms) == 0:
        return file_name, 0

    pts_query_ms_valid = np.asarray(pts_query_ms_valid)
    pts_local_ps = model_space_to_patch_space_list(
        pts_to_convert_ms=pts_local_ms,
        pts_patch_center_ms=pts_query_ms_valid,
        patch_radius_ms=patch_radius,
        numerical_stability_z_factor=10.0,
    )

    render_count = 0
    for i, pts_ps in enumerate(tqdm(pts_local_ps, desc='Rendering images', leave=False)):
        query_abs_id = query_abs_ids_valid[i]
        for method in pts_to_img_methods:
            _ = pts_to_img_cached(
                pts_ps_xy=pts_ps[:, :2],
                pts_data=pts_ps[:, 2],
                resolution=hm_interp_size,
                method=method,
                cache_dir=cache_dir,
                context_radius_factor=context_radius_factor,
                # meters_per_pixel: see ipes_img_data_loader.py's _cache_key for why this must be
                # part of the key too (it affects patch_radius, and therefore which points get
                # resolved for a given query id, same as resolution/context_radius_factor)
                cache_key='{}_{}_hm_{}'.format(cache_key_prefix, query_abs_id, meters_per_pixel),
            )
            render_count += 1

        for method in rgb_to_img_methods:
            _ = pts_to_img_cached(
                pts_ps_xy=pts_ps[:, :2],
                pts_data=pts_local_rgb[i],
                resolution=hm_interp_size,
                method=method,
                cache_dir=cache_dir,
                context_radius_factor=context_radius_factor,
                cache_key='{}_{}_rgb_{}'.format(cache_key_prefix, query_abs_id, meters_per_pixel),
            )
            render_count += 1

    return file_name, render_count


def precompute_img_cache_for_fit(
        in_file: str,
        train_set: str,
        val_set: str,
        hm_interp_size: int,
        hm_size: int,
        context_radius_factor: float,
        meters_per_pixel: float,
        dataset_step: int,
        pts_to_img_methods: typing.List[str],
        rgb_to_img_methods: typing.List[str],
        refresh_cache: bool):
    from source.dataloaders.base_data_module import get_dataset_dir

    if os.path.splitext(in_file)[1].lower() != '.txt':
        print(f'Skipping img_cache precompute for non-dataset input: {in_file}')
        return

    dataset_dir = get_dataset_dir(in_file)
    cache_dir = os.path.join(dataset_dir, 'img_cache')

    shape_names_raw = _read_shape_list(train_set) + _read_shape_list(val_set)
    tasks_expanded = _expand_shape_ranges(shape_names_raw=shape_names_raw, dataset_step=dataset_step)
    if len(tasks_expanded) == 0:
        print('No cache precompute tasks found.')
        return

    # Deduplicate exact windows that may appear in both train/val files.
    tasks_expanded = list(dict.fromkeys(tasks_expanded))

    # Marker keyed by exactly which (shape, id-range) tasks this run needs -- NOT just "does the
    # shared img_cache dir have any files in it". img_cache is one flat, hash-keyed directory
    # shared across every shape/dataset ever used from this dataset_dir (see pts_to_img_cached),
    # so a directory-non-empty check skips precompute for a run whose shapes/ranges were never
    # actually rendered, as long as *some other* run had already populated the directory. That
    # silently pushed all of this run's rendering onto the DataLoader workers instead, one
    # sample at a time inside the training loop with no batch parallelism -- on a 100x-densified
    # shape (ca_13_cp100) this stalled a fit run for 64+ hours on 3 of 388 steps before
    # being caught. The marker file's name is a hash of the exact task list, so any new shape or
    # id-range invalidates it and forces a real precompute; unchanged re-runs of the same config
    # still skip fast.
    from source.base.fs import str_to_consistent_hash
    tasks_fingerprint = str_to_consistent_hash(repr(sorted(tasks_expanded)))
    done_marker = os.path.join(cache_dir, '.precompute_done_{}'.format(tasks_fingerprint))

    if refresh_cache and os.path.exists(cache_dir):
        print(f'Refreshing img_cache at {cache_dir}')
        shutil.rmtree(cache_dir)
    elif os.path.exists(done_marker):
        print(f'img_cache already covers these {len(tasks_expanded)} tasks (marker {done_marker} found), skipping precompute.')
        return

    os.makedirs(cache_dir, exist_ok=True)

    tasks = [
        {
            'dataset_dir': dataset_dir,
            'hm_interp_size': hm_interp_size,
            'hm_size': hm_size,
            'context_radius_factor': context_radius_factor,
            'meters_per_pixel': meters_per_pixel,
            'pts_to_img_methods': pts_to_img_methods,
            'rgb_to_img_methods': rgb_to_img_methods,
            'file_name': file_name,
            'start_id': start_id,
            'end_id': end_id,
        }
        for (file_name, start_id, end_id) in tasks_expanded
    ]

    # Capped well below os.cpu_count() (32 on this machine): each task independently mmaps its
    # shape's chunkPoints and allocates a fresh (not in-place -- the source array is a read-only
    # mmap) normalized RGB copy, which for a 100x-densified shape like ca_13_cp100 is ~800MB
    # per task. With chunksize=1, imap_unordered gives no guarantee tasks for the same dense
    # shape are spread out over time, so all cpu_count() workers can end up doing this
    # concurrently -- that's what caused an ArrayMemoryError here despite tens of GB of system
    # RAM being free (transient concurrent allocation pressure, not a true system-wide OOM).
    num_workers = min(8, max(1, int(os.cpu_count() or 1)))
    print(f'Precomputing img_cache using {num_workers} worker processes for {len(tasks)} tasks...')

    try:
        with mp.get_context('spawn').Pool(processes=num_workers) as pool:
            total_renders = 0
            for file_name, render_count in tqdm(pool.imap_unordered(_task_worker, tasks, chunksize=1), total=len(tasks), desc='Precomputing img_cache'):
                total_renders += render_count
    except KeyboardInterrupt:
        print('img_cache precompute interrupted by user.')
        raise

    print(f'img_cache precompute finished: {len(tasks)} tasks, renders={total_renders}')
    with open(done_marker, 'w') as f:
        f.write('{} tasks\n'.format(len(tasks)))
