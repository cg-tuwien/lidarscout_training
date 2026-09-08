"""
Precomputes and caches KDTree radius-query results (which points are local to each query patch)
once per shape, as plain indices -- so IpesDataset's DataLoader workers never need to load a
shape's full point cloud or build their own KDTree.

Why this exists: PyTorch's DataLoader hands out dataset indices to num_workers worker processes
round-robin (worker w gets items w, w+num_workers, w+2*num_workers, ...). Windows for one shape
are consecutive indices (see IpesDataset.get_shape_names), so any shape with at least
num_workers windows gets touched by every worker -- each of which was independently loading the
full point cloud and building its own KDTree (source/dataloaders/ipes_data_loader.py's
`kdtree_cache` only avoids rebuilding *within* one worker process, not across them). For a shape
with tens of millions of points this is what exhausted memory (N workers x one full KDTree each).

The radius query itself doesn't need to happen more than once per shape: it depends only on the
tree, each query point, and a radius that's fixed by config (context_radius_factor,
hm_interp_size, meters_per_pixel), not on which window or epoch asks for it. So this precomputes
it once (single process, one KDTree, one batched query using scipy's own internal
multithreading -- query_ball_point(workers=-1) releases the GIL in C, no multiprocessing needed
here), and caches the result as a CSR-style (indices, offsets) pair: offsets[i]:offsets[i+1]
slices `indices` for query point i's local point ids. Cached in the exact same fixed order as
heightmaps.bin / heightmaps_query.npy, so a window's [start_id:end_id] slice lines up directly --
no coordinate matching needed.

Deliberately caches *indices*, not resolved xyz/rgb: query points aren't guaranteed to be
spatially separated (they're drawn randomly, only sorted by x for train/val splitting), so
nearby query points' local point sets can overlap heavily -- caching resolved coordinates would
duplicate that overlap on disk for every window that shares it. Indices reference the shared
point array instead, so there's no duplication, and resolving them is a cheap fancy-index once
the caller has the (now much cheaper to load, see source/base/fs.py's load_csv_points_cached)
point array in memory.
"""
import os
import typing

import numpy as np


def get_patch_radius(hm_interp_size: int, context_radius_factor: float, meters_per_pixel: float) -> float:
    """Must match IpesDataset._get_patch_radius_p2(hm_res=hm_interp_size) exactly -- this is the
    radius local-subsample queries use ("local patch must be after augmentation" in
    ipes_data_loader.py, i.e. the larger interpolation-context radius, not the smaller GT-only
    one)."""
    hm_diagonal = np.sqrt(2.0) * meters_per_pixel * hm_interp_size
    return hm_diagonal * 0.5 * context_radius_factor


def local_points_cache_paths(dataset_dir: str, shape_name: str, patch_radius: float) -> typing.Tuple[str, str]:
    """
    Returns (indices_path, offsets_path) -- two plain .npy files, not one .npz. This matters:
    .npz is a zip archive and can't be memory-mapped, so *every* reader would have to fully load
    it. For a dense/large shape, `indices` alone can be many GB (local point sets overlap heavily
    since query points aren't spatially separated -- see module docstring), so on a real
    100x-densified test shape this file was 11.2 GB; loading that fully in every DataLoader
    worker would have been worse than the KDTree duplication this cache exists to avoid. Plain
    .npy supports mmap_mode='r', so slice_local_point_ids() only pages in the bytes it actually
    touches, and separate worker processes mmap-ing the same file share physical pages via the OS
    page cache (same mechanism as source/base/fs.py's load_csv_points_cached). `offsets` is tiny
    (num_query_points + 1 int64s) so it's always loaded fully, no mmap needed for it.

    folding patch_radius into the filename means changing context_radius_factor / hm_interp_size
    / meters_per_pixel naturally invalidates the cache, no separate metadata bookkeeping needed.

    GT-mode (fit/test) only: reconstruction (predict) reconstructs a given point cloud once, so
    it isn't routed through this disk-persisted cache at all -- see
    IpesDataset._get_local_subsamples_fixed_radius_indexed.
    """
    from source.base.fs import str_to_consistent_hash

    radius_key = str_to_consistent_hash(f'{patch_radius:.6f}')
    base_dir = os.path.join(dataset_dir, 'cache_local_pts', shape_name)
    return os.path.join(base_dir, f'indices_{radius_key}.npy'), os.path.join(base_dir, f'offsets_{radius_key}.npy')


def build_local_point_index_cache(
        chunk_pts_xy: np.ndarray, query_pts_xy: np.ndarray, patch_radius: float,
        indices_out_path: str, batch_size: int = 500
) -> np.ndarray:
    """
    One KDTree build, then the radius query in small batches, two passes, writing the (possibly
    huge) result straight to a memory-mapped output file instead of an in-process array.

    A single all-at-once query_ball_point(workers=-1) call for every query point in a shape
    turned out to be a memory bottleneck on its own, not (as an earlier version of this function
    assumed) how the results were then copied around in Python: on a real 35M-point/10k-query
    test shape with heavily overlapping query patches (see module docstring), the query call
    itself peaked at ~46-49GB regardless of how its output was post-processed -- that memory is
    scipy's own, spent before any of this function's Python code runs. Batching the query (below)
    fixed that. But the *unavoidable* final result for that same shape is itself large enough
    (tens of GB) that even holding it as a single in-process array before writing it out was too
    much against a tight RAM budget -- so this writes directly into a memory-mapped .npy file
    (np.lib.format.open_memmap) sized exactly once pass 1 determines it, batch by batch. The OS
    pages writes to disk as needed instead of this process's private memory ever holding the
    whole thing; peak RAM is one batch's worth of query results, not the total result size.

    Returns `offsets` (tiny, num_query_points + 1 int64s -- kept in memory, no need to mmap it).
    `indices_out_path` is written to directly, not via a caller-visible return value; the caller
    is expected to os.replace() it into its final location once this returns.
    """
    from source.base.proximity import make_kdtree, query_ball_kdtree

    kdtree = make_kdtree(chunk_pts_xy, lib='scipy')
    n = query_pts_xy.shape[0]

    # pass 1: counts only
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        id_lists = query_ball_kdtree(
            kdtree=kdtree, pts_query=query_pts_xy[start:end], r=patch_radius, workers=-1, return_sorted=False)
        for i, ids in enumerate(id_lists):
            counts[start + i] = len(ids)
        del id_lists

    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    # pass 2: fill the exactly-sized, memory-mapped output directly, batch by batch
    indices_mmap = np.lib.format.open_memmap(indices_out_path, mode='w+', dtype=np.int64, shape=(int(offsets[-1]),))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        id_lists = query_ball_kdtree(
            kdtree=kdtree, pts_query=query_pts_xy[start:end], r=patch_radius, workers=-1, return_sorted=True)
        for i, ids in enumerate(id_lists):
            qi = start + i
            indices_mmap[offsets[qi]:offsets[qi + 1]] = ids
        del id_lists
    indices_mmap.flush()
    del indices_mmap  # release the writable mapping so the file can be moved/reopened afterward

    return offsets


def load_or_build_local_point_index_cache(
        dataset_dir: str, shape_name: str, chunk_pts_source_file: str,
        get_chunk_pts_xy: typing.Callable[[], np.ndarray], get_query_pts_xy: typing.Callable[[], np.ndarray],
        patch_radius: float, mmap_indices: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    get_chunk_pts_xy/get_query_pts_xy are callables, not arrays -- so a (possibly huge) point
    cloud is only ever loaded if the cache actually needs to be (re)built. A warm cache never
    touches it.

    mmap_indices=True (the default, and what every DataLoader-facing caller should use) opens
    `indices` via mmap_mode='r' instead of fully loading it -- see local_points_cache_paths()'s
    docstring for why that matters. Only set it False for callers that are about to consume the
    whole array anyway (there currently are none in this codebase; kept as an explicit escape
    hatch rather than silently always mmap-ing).
    """
    from source.base.fs import call_necessary, make_dir_for_file

    indices_path, offsets_path = local_points_cache_paths(dataset_dir, shape_name, patch_radius)
    if not call_necessary(file_in=chunk_pts_source_file, file_out=[indices_path, offsets_path]):
        try:
            indices = np.load(indices_path, mmap_mode='r' if mmap_indices else None)
            offsets = np.load(offsets_path)
            return indices, offsets
        except Exception:
            pass  # fall through and rebuild if the cache is corrupt/truncated

    make_dir_for_file(indices_path)
    tmp_indices_path = f'{indices_path}.tmp.{os.getpid()}.npy'
    tmp_offsets_path = f'{offsets_path}.tmp.{os.getpid()}.npy'
    # build_local_point_index_cache writes indices straight to tmp_indices_path via a writable
    # mmap (see its docstring) -- offsets is the only array actually returned in memory
    offsets = build_local_point_index_cache(get_chunk_pts_xy(), get_query_pts_xy(), patch_radius, tmp_indices_path)
    np.save(tmp_offsets_path, offsets)
    # offsets first: a reader that sees offsets but not yet indices will just cache-miss (call_necessary
    # checks both files exist) and rebuild, rather than reading a mismatched pair
    os.replace(tmp_offsets_path, offsets_path)
    os.replace(tmp_indices_path, indices_path)
    indices = np.load(indices_path, mmap_mode='r' if mmap_indices else None)
    return indices, offsets


def slice_local_point_ids(
        indices: np.ndarray, offsets: np.ndarray, start_id: int, end_id: int) -> typing.List[np.ndarray]:
    end_id = min(end_id, offsets.shape[0] - 1)
    return [indices[offsets[i]:offsets[i + 1]] for i in range(start_id, end_id)]


def _shape_source_files(dataset_dir: str, file_name: str) -> typing.Tuple[str, str]:
    chunk_pts_file = os.path.join(dataset_dir, 'bins', file_name, 'chunkPoints.csv')
    query_pts_file = os.path.join(dataset_dir, 'cache_gt', file_name, 'heightmaps_query.npy')
    return chunk_pts_file, query_pts_file


def precompute_one_shape(dataset_dir: str, file_name: str, patch_radius: float) -> bool:
    """Returns True if a (re)build actually happened, False if the cache was already warm or the
    shape has no GT data to precompute for (e.g. a reconstruction-only input)."""
    from source.base.fs import call_necessary, load_csv_points_cached

    chunk_pts_file, query_pts_file = _shape_source_files(dataset_dir, file_name)
    if not os.path.exists(chunk_pts_file) or not os.path.exists(query_pts_file):
        return False

    indices_path, offsets_path = local_points_cache_paths(dataset_dir, file_name, patch_radius)
    if not call_necessary(file_in=[chunk_pts_file, query_pts_file], file_out=[indices_path, offsets_path]):
        return False

    load_or_build_local_point_index_cache(
        dataset_dir=dataset_dir, shape_name=file_name, chunk_pts_source_file=chunk_pts_file,
        get_chunk_pts_xy=lambda: load_csv_points_cached(chunk_pts_file)[:, :2],
        get_query_pts_xy=lambda: np.asarray(np.load(query_pts_file)[:, :2]),
        patch_radius=patch_radius)
    return True


def precompute_local_points_cache(
        in_files: typing.List[str], hm_interp_size: int, context_radius_factor: float, meters_per_pixel: float,
        show_progress: bool = True):
    """
    Explicit precompute entry point, meant to run once in the main process before any DataLoader
    worker exists (mirrors img_cache_precompute.py's precompute_img_cache_for_fit) -- avoids the
    N-way redundant build that the lazy fallback in ipes_data_loader.py would otherwise risk if
    several workers hit a cold cache for the same shape at once.
    """
    from tqdm import tqdm
    from source.dataloaders.base_data_module import in_file_is_dataset, get_dataset_dir, read_shape_list

    patch_radius = get_patch_radius(hm_interp_size, context_radius_factor, meters_per_pixel)

    dataset_dir = None
    shape_names_raw = []
    for in_file in in_files:
        if not in_file or not in_file_is_dataset(in_file) or not os.path.exists(in_file):
            continue
        dataset_dir = get_dataset_dir(in_file)
        shape_names_raw += read_shape_list(in_file)
    shape_names_raw = sorted(set(shape_names_raw))
    if dataset_dir is None or not shape_names_raw:
        return

    iterator = tqdm(shape_names_raw, desc='Precomputing local-points cache') if show_progress else shape_names_raw
    for shape_entry in iterator:
        file_name = shape_entry.split(',')[0]
        precompute_one_shape(dataset_dir, file_name, patch_radius)


def _unit_test_local_points_cache():
    """Cross-checks the cached (indices, offsets) against a direct per-query-point KDTree ball
    query, and checks that slice_local_point_ids matches build_local_point_index_cache's own
    per-query-point breakdown exactly. Also exercises batch_size < num_query_points, since the
    default is much larger than this test's tiny query count."""
    import tempfile
    from source.base.proximity import make_kdtree, query_ball_kdtree

    rng = np.random.default_rng(0)
    chunk_pts_xy = rng.uniform(0.0, 1000.0, size=(2000, 2))
    query_pts_xy = rng.uniform(0.0, 1000.0, size=(37, 2))
    patch_radius = 80.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        indices_path = os.path.join(tmp_dir, 'indices.npy')
        offsets = build_local_point_index_cache(
            chunk_pts_xy, query_pts_xy, patch_radius, indices_path, batch_size=5)
        indices = np.load(indices_path)
        assert offsets.shape[0] == query_pts_xy.shape[0] + 1
        assert offsets[-1] == indices.shape[0]

        kdtree = make_kdtree(chunk_pts_xy, lib='scipy')
        reference_lists = query_ball_kdtree(
            kdtree=kdtree, pts_query=query_pts_xy, r=patch_radius, workers=1, return_sorted=True)

        sliced = slice_local_point_ids(indices, offsets, 0, query_pts_xy.shape[0])
        # a partial window slice must match the corresponding sub-range exactly
        partial = slice_local_point_ids(indices, offsets, 10, 20)
        total_hits = int(indices.shape[0])
        del indices  # release before the TemporaryDirectory cleans up (Windows can't delete an open mmap)

    for i, (cached_ids, ref_ids) in enumerate(zip(sliced, reference_lists)):
        assert np.array_equal(np.sort(cached_ids), np.sort(np.asarray(ref_ids, dtype=np.int64))), \
            f'query point {i}: cached local point ids differ from a direct ball query'

    assert len(partial) == 10
    for i in range(10):
        assert np.array_equal(np.sort(partial[i]), np.sort(sliced[10 + i]))

    print('_unit_test_local_points_cache: OK ({} query points, {} points, {} total local hits)'.format(
        query_pts_xy.shape[0], chunk_pts_xy.shape[0], total_hits))


def _unit_test_precompute_roundtrip():
    """End-to-end: precompute_one_shape() writes a cache file that
    load_or_build_local_point_index_cache() then reads back without rebuilding, and a second
    precompute call is a no-op given unchanged inputs."""
    import tempfile
    from source.base.fs import make_dir_for_file

    rng = np.random.default_rng(1)
    chunk_pts = rng.uniform(0.0, 500.0, size=(300, 3))
    query_pts = rng.uniform(0.0, 500.0, size=(15, 3))

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_pts_file = os.path.join(tmp_dir, 'bins', 'shapeA', 'chunkPoints.csv')
        query_pts_file = os.path.join(tmp_dir, 'cache_gt', 'shapeA', 'heightmaps_query.npy')
        make_dir_for_file(chunk_pts_file)
        make_dir_for_file(query_pts_file)
        np.savetxt(chunk_pts_file, np.concatenate([chunk_pts, np.zeros((300, 3))], axis=1), delimiter=',')
        np.save(query_pts_file, query_pts)

        patch_radius = get_patch_radius(hm_interp_size=64, context_radius_factor=1.5, meters_per_pixel=10.0)

        built = precompute_one_shape(tmp_dir, 'shapeA', patch_radius)
        assert built, 'first precompute call should build the cache'
        indices_path, offsets_path = local_points_cache_paths(tmp_dir, 'shapeA', patch_radius)
        assert os.path.exists(indices_path) and os.path.exists(offsets_path)

        built_again = precompute_one_shape(tmp_dir, 'shapeA', patch_radius)
        assert not built_again, 'second precompute call should be a no-op (cache already warm)'

        indices, offsets = load_or_build_local_point_index_cache(
            dataset_dir=tmp_dir, shape_name='shapeA', chunk_pts_source_file=chunk_pts_file,
            get_chunk_pts_xy=lambda: (_ for _ in ()).throw(AssertionError('should not need to load points, cache is warm')),
            get_query_pts_xy=lambda: (_ for _ in ()).throw(AssertionError('should not need to load points, cache is warm')),
            patch_radius=patch_radius)
        assert offsets.shape[0] == query_pts.shape[0] + 1
        assert isinstance(indices, np.memmap), 'indices should be memory-mapped by default, not fully loaded'
        # Windows can't delete a file that's still mmap'd (no FILE_SHARE_DELETE) -- release
        # explicitly before the TemporaryDirectory context tries to clean up
        del indices
    print('_unit_test_precompute_roundtrip: OK')


if __name__ == '__main__':
    _unit_test_local_points_cache()
    _unit_test_precompute_roundtrip()
