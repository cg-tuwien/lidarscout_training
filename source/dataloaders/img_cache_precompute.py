import os
import shutil
import typing
import multiprocessing as mp

import numpy as np
from scipy.spatial import KDTree

from source.base.normalization import model_space_to_patch_space_list
from source.base.point_cloud import pts_to_img_cached


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

    if os.path.exists(query_pts_file_cache) and os.path.exists(hm_file_cache):
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
    pts_file = os.path.join(dataset_dir, 'bins', file_name, 'chunkPoints.csv')
    chunk_pts_all = np.loadtxt(pts_file, dtype=np.float64, delimiter=',')
    has_colors = chunk_pts_all.shape[1] == 6
    chunk_pts_xyz = chunk_pts_all[:, :3]
    chunk_pts_rgb = chunk_pts_all[:, 3:6] if has_colors else np.full(chunk_pts_xyz.shape, np.nan)
    if file_name in ['swisssurface3d']:
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

    patch_radius = np.sqrt(2.0) * meters_per_pixel * hm_interp_size * 0.5 * context_radius_factor
    kdtree = KDTree(chunk_pts_ms[:, :2], leafsize=1000)
    patch_pts_ids_lists = kdtree.query_ball_point(
        x=pts_query_ms[:, :2], r=patch_radius, workers=1, return_sorted=True)

    min_point_count = 100
    pts_local_ms = []
    pts_local_rgb = []
    pts_query_ms_valid = []
    for i, ids in enumerate(patch_pts_ids_lists):
        if len(ids) > min_point_count:
            ids_np = np.asarray(ids)
            pts_local_ms.append(chunk_pts_ms[ids_np])
            pts_local_rgb.append(chunk_pts_rgb[ids_np])
            pts_query_ms_valid.append(pts_query_ms[i])

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
    for i, pts_ps in enumerate(pts_local_ps):
        for method in pts_to_img_methods:
            _ = pts_to_img_cached(
                pts_ps_xy=pts_ps[:, :2],
                pts_data=pts_ps[:, 2],
                resolution=hm_interp_size,
                method=method,
                cache_dir=cache_dir,
            )
            render_count += 1

        for method in rgb_to_img_methods:
            _ = pts_to_img_cached(
                pts_ps_xy=pts_ps[:, :2],
                pts_data=pts_local_rgb[i],
                resolution=hm_interp_size,
                method=method,
                cache_dir=cache_dir,
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
    cache_exists = os.path.exists(cache_dir) and any(os.scandir(cache_dir))

    if cache_exists and not refresh_cache:
        print(f'img_cache exists at {cache_dir}, skipping precompute (use --refresh_cache True to rebuild).')
        return

    if refresh_cache and os.path.exists(cache_dir):
        print(f'Refreshing img_cache at {cache_dir}')
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    shape_names_raw = _read_shape_list(train_set) + _read_shape_list(val_set)
    tasks_expanded = _expand_shape_ranges(shape_names_raw=shape_names_raw, dataset_step=dataset_step)
    if len(tasks_expanded) == 0:
        print('No cache precompute tasks found.')
        return

    # Deduplicate exact windows that may appear in both train/val files.
    tasks_expanded = list(dict.fromkeys(tasks_expanded))

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

    num_workers = max(1, int(os.cpu_count() or 1))
    print(f'Precomputing img_cache using {num_workers} worker processes for {len(tasks)} tasks...')

    try:
        with mp.get_context('spawn').Pool(processes=num_workers) as pool:
            done = 0
            total_renders = 0
            for file_name, render_count in pool.imap_unordered(_task_worker, tasks, chunksize=1):
                done += 1
                total_renders += render_count
                if done % 10 == 0 or done == len(tasks):
                    print(f'img_cache precompute: {done}/{len(tasks)} tasks, renders={total_renders}, last={file_name}')
    except KeyboardInterrupt:
        print('img_cache precompute interrupted by user.')
        raise

    print(f'img_cache precompute finished: {len(tasks)} tasks, renders={total_renders}')
