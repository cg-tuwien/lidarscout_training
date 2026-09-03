"""
Python port of simlod's dataset-generation tools, so LidarScout training data can be produced
directly from a folder of LAS/LAZ tiles without a separate C++ build.

Ports (and lightly cleans up) two tools from simlod:
  - src/main_create_training.cpp: for a set of random query points, bins ALL points from
    intersecting tiles into per-query heightmap grids (mean height per pixel) plus matching
    mean-RGB grids. Written to heightmaps.bin / rgb_{0,1,2}.bin, binary-compatible with
    source/dataloaders/ipes_data_loader.py's reader (dtype '3f8,(N,N)f4' per record: query
    xyz as float64, followed by an NxN float32 grid, row-major with pixelID = px + N * py).
  - tools/heightmap_filter.mjs: writes a sparse "chunkPoints.csv" (x, y, z, r, g, b) by taking
    one point out of every `chunk_points_stride` -- by default 50_000, which is also LASzip's
    default point-compression chunk size, so this is effectively "decode the first point of
    every LAZ compression chunk", the cheapest possible spatially-distributed subsample.

Deviations from the C++ tool (deliberate cleanups, see plan/notes):
  - Pixel binning uses `np.floor(N * u)` instead of C++'s `int(N * u)` (truncation towards
    zero). For points just outside a heightmap window on its negative edge (u in (-1/N, 0)),
    C++'s truncation maps them to pixel 0 instead of correctly excluding them -- a narrow
    (~one pixel wide) contamination strip on the bottom/left edge of every ground-truth
    heightmap. `floor` is the standard, symmetric convention and is what the rest of this
    repo's grid code (see source/base/point_cloud.py) already assumes.
  - Query point sampling draws points per-tile via `numpy`'s multinomial + choice instead of
    C++'s global reservoir of random indices; statistically equivalent for
    num_query_points << num_points_total (true here by many orders of magnitude), and avoids
    needing seekable point-index access into compressed LAZ files.

Memory: tiles are read and processed one at a time (matching the C++ tool's design), so peak
memory is bounded by the largest single tile, not the size of the whole dataset. Point clouds
of tens to hundreds of GB are safe to process as long as no single input tile is enormous.
"""
import dataclasses
import os
import typing

import numpy as np


LAS_EXTENSIONS = ('.las', '.laz')


@dataclasses.dataclass
class TileInfo:
    path: str
    mins: np.ndarray  # (3,) float64, xyz
    maxs: np.ndarray  # (3,) float64, xyz
    num_points: int


def list_las_files(directory: str) -> typing.List[str]:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(LAS_EXTENSIONS))


def gather_tile_info(files: typing.List[str], show_progress: bool = True) -> typing.List[TileInfo]:
    """Read only headers (bbox + point count), never point data. Cheap even for thousands of tiles."""
    import laspy
    from tqdm import tqdm

    infos = []
    iterator = tqdm(files, desc='Reading tile headers') if show_progress else files
    for f in iterator:
        with laspy.open(f) as reader:
            h = reader.header
            if h.point_count == 0:
                continue
            infos.append(TileInfo(
                path=f,
                mins=np.array(h.mins, dtype=np.float64),
                maxs=np.array(h.maxs, dtype=np.float64),
                num_points=int(h.point_count),
            ))
    return infos


def _read_tile_points(path: str) -> tuple:
    """Read one tile fully into memory (matches the C++ tool's per-tile loading). Returns
    (xyz float64 (k,3), rgb uint8 (k,3) or None if the format has no color)."""
    import laspy

    with laspy.open(path) as reader:
        las = reader.read()
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    if hasattr(las, 'red'):
        # LAS RGB is commonly stored as 16-bit; downscale to 8-bit only if it looks 16-bit,
        # matching main_create_training.cpp's `rgb[c] > 255 ? rgb[c] / 256 : rgb[c]`
        def _to_u8(channel):
            channel = np.asarray(channel).astype(np.int64)
            return np.where(channel > 255, channel // 256, channel).astype(np.uint8)
        rgb = np.column_stack([_to_u8(las.red), _to_u8(las.green), _to_u8(las.blue)])
    else:
        rgb = None
    return xyz, rgb


def _resolve_points_by_index(path: str, sorted_local_indices: np.ndarray) -> np.ndarray:
    """
    Read a handful of specific points out of a tile by index, via LasReader.seek() -- NOT a full
    tile decode. Benchmarked on a real 14.4M-point CA13_SAN_SIM tile: ~16ms per seek+read vs.
    ~1-6s to decode the whole tile, so for the "a few points per tile" access pattern this is
    dramatically cheaper than reading everything and indexing in memory. Indices should be sorted
    ascending -- seeking forward is the pattern this was benchmarked with.
    """
    import laspy

    pts = np.empty((len(sorted_local_indices), 3), dtype=np.float64)
    with laspy.open(path) as reader:
        for row, idx in enumerate(sorted_local_indices):
            reader.seek(int(idx))
            p = reader.read_points(1)
            pts[row] = (p.x[0], p.y[0], p.z[0])
    return pts


def pick_query_points(
        tiles: typing.List[TileInfo], num_query_points: int, rng: np.random.Generator,
        show_progress: bool = True) -> np.ndarray:
    """
    Pick `num_query_points` random points, spread across tiles proportional to their point count
    (statistically equivalent to picking uniformly among all points globally), and resolve their
    xyz via `_resolve_points_by_index` -- targeted seeks, not a full tile decode. This avoids an
    entire extra full-decode pass over the dataset just to pick query points (see module
    docstring / plan notes on the 3-full-passes -> ~1-full-pass restructuring).

    Points are sorted by x before returning, so that a simple index-range split (as
    datasets/laz_minimal/*.txt already do, e.g. test=[0,300) / train=[300,10000)) gives a
    spatially coherent train/val separation instead of a spatially interleaved one.
    """
    from tqdm import tqdm

    counts = np.array([t.num_points for t in tiles], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError('No points found in the input tiles.')
    probs = counts / total
    alloc = rng.multinomial(num_query_points, probs)

    def _resolve_for_tiles(tile_alloc_pairs, desc):
        pts = []
        iterator = tqdm(tile_alloc_pairs, desc=desc) if show_progress else tile_alloc_pairs
        for tile, k in iterator:
            if k == 0:
                continue
            k = min(int(k), tile.num_points)
            idx = np.sort(rng.choice(tile.num_points, size=k, replace=False))
            pts.append(_resolve_points_by_index(tile.path, idx))
        return pts

    query_pts = _resolve_for_tiles(list(zip(tiles, alloc)), desc='Picking query points')
    result = np.concatenate(query_pts, axis=0) if query_pts else np.empty((0, 3))

    # multinomial rounding can occasionally leave us a few points short (tiles with fewer
    # points than allocated); top up with extra draws from the largest tile if necessary
    if result.shape[0] < num_query_points:
        biggest = max(tiles, key=lambda t: t.num_points)
        missing = num_query_points - result.shape[0]
        extra = _resolve_for_tiles([(biggest, min(missing, biggest.num_points))], desc='Topping up query points')
        result = np.concatenate([result] + extra, axis=0)

    result = result[:num_query_points]
    result = result[np.argsort(result[:, 0], kind='stable')]
    return result


def _accumulate_tile_into(
        tile: TileInfo, q_min_xy: np.ndarray, q_max_xy: np.ndarray,
        heightmap_size: int, heightmap_world: float, max_windows_per_batch: int,
        count_flat: np.ndarray, z_sum_flat: np.ndarray, rgb_sum_flat: np.ndarray, s2: int,
        want_chunk_points: bool, chunk_size: int, points_per_chunk: int):
    """
    Reads one tile and accumulates its contribution directly into the given flat (n*S*S,)
    count/z_sum/(3,n*S*S) rgb_sum arrays, mutated in place. Each query window's contribution is
    added immediately (np.add.at) rather than collected across the whole tile first, so peak
    extra memory is bounded by one batch of `max_windows_per_batch` windows -- NOT by how many
    windows or points the tile ends up touching in total. (An earlier version of this function
    collected every window's hits into a growing list and concatenated once per tile; on a real,
    dense CA13_SAN_SIM tile with ~240 affected windows that tried to allocate a 4.4 GiB array.
    Don't reintroduce that pattern.)

    Returns (any_rgb_seen: bool, chunk_lines: str | None).
    """
    intersects = (
        (q_max_xy[:, 0] > tile.mins[0]) & (q_min_xy[:, 0] < tile.maxs[0]) &
        (q_max_xy[:, 1] > tile.mins[1]) & (q_min_xy[:, 1] < tile.maxs[1]))
    affected = np.nonzero(intersects)[0]
    if affected.shape[0] == 0 and not want_chunk_points:
        return False, None

    xyz, rgb = _read_tile_points(tile.path)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    any_rgb = rgb is not None

    chunk_lines = _format_chunk_points_for_tile(xyz, rgb, chunk_size, points_per_chunk) if want_chunk_points else None

    for batch_start in range(0, affected.shape[0], max_windows_per_batch):
        batch = affected[batch_start:batch_start + max_windows_per_batch]

        # (b, num_points) broadcast: one row per query window in this batch
        px_all = np.floor((x[None, :] - q_min_xy[batch, 0, None]) / heightmap_world * heightmap_size)
        py_all = np.floor((y[None, :] - q_min_xy[batch, 1, None]) / heightmap_world * heightmap_size)
        valid_all = (px_all >= 0) & (px_all < heightmap_size) & (py_all >= 0) & (py_all < heightmap_size)
        pid_all = (px_all + heightmap_size * py_all).astype(np.int64)
        del px_all, py_all

        for row, qi in enumerate(batch):
            valid = valid_all[row]
            if not valid.any():
                continue
            flat_idx = qi * s2 + pid_all[row, valid]

            np.add.at(count_flat, flat_idx, 1)
            np.add.at(z_sum_flat, flat_idx, z[valid])
            if rgb is not None:
                crgb = rgb[valid]  # one gather for all 3 channels, not 3 separate strided ones
                for c in range(3):
                    np.add.at(rgb_sum_flat[c], flat_idx, crgb[:, c])
        del valid_all, pid_all
    del xyz, rgb, x, y, z
    return any_rgb, chunk_lines


# module-level (not instance) state for multiprocessing workers, set up by _init_worker() --
# spawned worker processes each get their own copy via the Pool initializer, not shared memory
_worker_state = {}


def _init_worker(q_min_xy, q_max_xy, heightmap_size, heightmap_world, max_windows_per_batch,
                 want_chunk_points, chunk_size, points_per_chunk, n, s2):
    _worker_state.update(
        q_min_xy=q_min_xy, q_max_xy=q_max_xy, heightmap_size=heightmap_size, heightmap_world=heightmap_world,
        max_windows_per_batch=max_windows_per_batch, want_chunk_points=want_chunk_points,
        chunk_size=chunk_size, points_per_chunk=points_per_chunk, n=n, s2=s2)


def _tile_chunk_worker(tile_chunk: typing.List[TileInfo]):
    """
    Processes a chunk of tiles (not just one) so a worker can maintain its own local accumulator
    across all of them and return it once, instead of returning per-tile contributions that would
    have to be collected somewhere -- keeps peak memory per worker fixed at
    O(num_query_points * S * S), the same as the serial path's single shared accumulator, instead
    of scaling with however much a busy chunk of tiles happens to touch.
    """
    s = _worker_state
    n_s2 = s['n'] * s['s2']
    local_count = np.zeros(n_s2, dtype=np.int64)
    local_z_sum = np.zeros(n_s2, dtype=np.float64)
    local_rgb_sum = np.zeros((3, n_s2), dtype=np.float64)
    any_rgb = False
    chunk_lines_parts = []

    for tile in tile_chunk:
        rgb_seen, lines = _accumulate_tile_into(
            tile, s['q_min_xy'], s['q_max_xy'], s['heightmap_size'], s['heightmap_world'],
            s['max_windows_per_batch'], local_count, local_z_sum, local_rgb_sum, s['s2'],
            s['want_chunk_points'], s['chunk_size'], s['points_per_chunk'])
        any_rgb = any_rgb or rgb_seen
        if lines:
            chunk_lines_parts.append(lines)

    return local_count, local_z_sum, local_rgb_sum, any_rgb, ''.join(chunk_lines_parts)


def build_heightmaps(
        tiles: typing.List[TileInfo], query_points_ms: np.ndarray,
        heightmap_size: int, pixel_size: float, show_progress: bool = True,
        max_windows_per_batch: int = 16,
        chunk_points_csv_path: typing.Optional[str] = None, chunk_size: int = 50_000,
        points_per_chunk: int = 1, num_workers: int = 1,
) -> tuple:
    """
    Bin all tile points into per-query-point heightmap grids (mean height per pixel) and
    matching mean-RGB grids. If `chunk_points_csv_path` is given, also writes the sparse
    "chunkPoints.csv" sample (see `write_chunk_points_csv`) from the same tile reads -- every
    tile needs a full decode for chunk-point sampling regardless of whether it affects any
    heightmap, so folding that into this loop avoids a second full-decode pass over the whole
    dataset just for that (see module docstring / plan notes: this was 3 full decode passes per
    tile -- query point picking, binning, chunk points -- now down to ~1).

    Performance note: pixel binning is relative to each query point's own window, so a tile with
    k affected query points needs k passes over that tile's points -- there's no way around that
    without a spatial index. What *is* avoidable is redoing that work with plain Python-level
    looping: this bins `max_windows_per_batch` windows at a time via a single broadcast
    ((min(k, max_windows_per_batch), num_points) comparison) instead of one full pass per window.
    Batched (not all-at-once) to bound peak memory for tiles with many affected query points.

    `num_workers` > 1 processes tiles in parallel via multiprocessing. Tiles are split into
    `4 * num_workers` chunks (for dynamic load balancing across tiles of very different sizes);
    each worker keeps its own local accumulator (same shape/size as the main one) across the
    tiles in a chunk and returns it once per chunk, which the main process sums in -- so peak
    memory is O(num_workers * num_query_points * S * S), not dependent on tile size or point
    density. Pick num_workers based on available RAM alongside CPU count.

    :return: (height (n, S, S) float32, rgb (3, n, S, S) float32, count (n, S, S) int32)
             NaN where a pixel has no points (height/rgb) / 0 (count).
    """
    from tqdm import tqdm
    from source.base.fs import make_dir_for_file

    n = query_points_ms.shape[0]
    heightmap_world = heightmap_size * pixel_size
    half = heightmap_world / 2.0
    q_min_xy = query_points_ms[:, :2] - half
    q_max_xy = query_points_ms[:, :2] + half
    s2 = heightmap_size * heightmap_size

    count = np.zeros(n * s2, dtype=np.int64)
    z_sum = np.zeros(n * s2, dtype=np.float64)
    rgb_sum = np.zeros((3, n * s2), dtype=np.float64)
    any_rgb = False

    want_chunk_points = chunk_points_csv_path is not None
    if want_chunk_points:
        make_dir_for_file(chunk_points_csv_path)
        f_chunk_points = open(chunk_points_csv_path, 'w')
    else:
        f_chunk_points = None

    try:
        if num_workers <= 1:
            iterator = tqdm(tiles, desc='Binning points + writing chunk points') if show_progress else tiles
            for tile in iterator:
                rgb_seen, lines = _accumulate_tile_into(
                    tile, q_min_xy, q_max_xy, heightmap_size, heightmap_world, max_windows_per_batch,
                    count, z_sum, rgb_sum, s2, want_chunk_points, chunk_size, points_per_chunk)
                any_rgb = any_rgb or rgb_seen
                if f_chunk_points is not None and lines:
                    f_chunk_points.write(lines)
        else:
            import multiprocessing as mp

            num_chunks = max(num_workers, min(len(tiles), num_workers * 4))
            boundaries = np.linspace(0, len(tiles), num_chunks + 1).astype(int)
            tile_chunks = [tiles[boundaries[i]:boundaries[i + 1]] for i in range(num_chunks)
                          if boundaries[i + 1] > boundaries[i]]

            ctx = mp.get_context('spawn')
            with ctx.Pool(
                    processes=num_workers, initializer=_init_worker,
                    initargs=(q_min_xy, q_max_xy, heightmap_size, heightmap_world, max_windows_per_batch,
                             want_chunk_points, chunk_size, points_per_chunk, n, s2)) as pool:
                iterator = pool.imap_unordered(_tile_chunk_worker, tile_chunks, chunksize=1)
                if show_progress:
                    iterator = tqdm(
                        iterator, total=len(tile_chunks),
                        desc=f'Binning points ({num_workers} workers, {len(tile_chunks)} chunks)')
                for local_count, local_z_sum, local_rgb_sum, rgb_seen, lines in iterator:
                    count += local_count
                    z_sum += local_z_sum
                    rgb_sum += local_rgb_sum
                    any_rgb = any_rgb or rgb_seen
                    if f_chunk_points is not None and lines:
                        f_chunk_points.write(lines)
    finally:
        if f_chunk_points is not None:
            f_chunk_points.close()

    count = count.reshape(n, s2)
    z_sum = z_sum.reshape(n, s2)
    rgb_sum = rgb_sum.reshape(3, n, s2)
    count_grid = count.reshape(n, heightmap_size, heightmap_size)
    height = np.where(count > 0, z_sum / np.maximum(count, 1), np.nan).astype(np.float32)
    height = height.reshape(n, heightmap_size, heightmap_size)

    if any_rgb:
        rgb_out = np.where(count[None] > 0, rgb_sum / np.maximum(count, 1)[None], np.nan).astype(np.float32)
        rgb_out = rgb_out.reshape(3, n, heightmap_size, heightmap_size)
    else:
        rgb_out = None

    return height, rgb_out, count_grid


def resort_heightmaps_bin_by_query_x(out_dir: str, heightmap_size: int = 64):
    """
    Post-hoc fix-up for heightmaps.bin / rgb_*.bin files that were written with query points in
    an arbitrary (not x-sorted) order -- re-sorts the records in place by query x, so a simple
    index-range split (as datasets/laz_minimal/*.txt already do) gives a spatially coherent
    train/val separation. pick_query_points() now sorts by construction, so this is only needed
    for datasets generated before that change (or regenerated data you don't want to rebin from
    scratch just to fix the ordering -- this only permutes already-computed records, no
    LAS/LAZ decoding involved, so it's fast regardless of dataset size).
    """
    dt = np.dtype('3f8, ({0},{0})f4'.format(heightmap_size))
    hm_path = os.path.join(out_dir, 'heightmaps.bin')
    hm_data = np.fromfile(hm_path, dtype=dt)
    order = np.argsort(hm_data['f0'][:, 0], kind='stable')
    if np.array_equal(order, np.arange(order.shape[0])):
        print(f'{hm_path}: already sorted by query x, nothing to do')
        return

    hm_data[order].tofile(hm_path)
    print(f'resorted {hm_path} ({order.shape[0]} records)')

    for c in range(3):
        rgb_path = os.path.join(out_dir, f'rgb_{c}.bin')
        if not os.path.exists(rgb_path):
            continue
        rgb_data = np.fromfile(rgb_path, dtype=dt)
        assert rgb_data.shape[0] == order.shape[0], f'{rgb_path} has a different record count than heightmaps.bin'
        rgb_data[order].tofile(rgb_path)
        print(f'resorted {rgb_path}')


def write_heightmaps_bin(path: str, query_points_ms: np.ndarray, height: np.ndarray):
    """Binary-compatible with IpesDataset.get_hm_bin_file's reader:
    dtype '3f8,(S,S)f4' per record (query xyz float64, then an SxS float32 grid)."""
    from source.base.fs import make_dir_for_file

    n, s, _ = height.shape
    dt = np.dtype('3f8, ({0},{0})f4'.format(s))
    out = np.zeros(n, dtype=dt)
    out['f0'] = query_points_ms
    out['f1'] = height
    make_dir_for_file(path)
    out.tofile(path)


def _format_chunk_points_for_tile(
        xyz: np.ndarray, rgb: typing.Optional[np.ndarray], chunk_size: int, points_per_chunk: int
) -> typing.Optional[str]:
    """
    Formats the first `points_per_chunk` points of every `chunk_size`-sized run of an already
    decoded tile's points, as 'x, y, z, r, g, b \\n' lines (2 decimals) -- matching the format
    source/dataloaders/ipes_data_loader.py already reads via np.loadtxt(..., delimiter=',').
    Returns a single string (not written directly) so it can be sent back from a multiprocessing
    worker for the main process to write, as well as used for direct in-process writing.

    Port of tools/heightmap_filter.mjs (points_per_chunk=1, the default): `chunk_size` defaults
    to LASzip's default compression chunk size, so points_per_chunk=1 conceptually reads only the
    first point of each LAZ chunk. points_per_chunk > 1 generalizes this to "LOD-less LOD"
    sampling: the first N points of every chunk instead of just one -- denser, but spatially
    *clustered* rather than spatially uniform (see plan/notes on the LOD-less LOD dataset).
    """
    chunk_starts = np.arange(0, xyz.shape[0], chunk_size)
    if chunk_starts.shape[0] == 0:
        return None
    idx = np.concatenate([
        np.arange(start, min(start + points_per_chunk, xyz.shape[0])) for start in chunk_starts])
    xyz_s = xyz[idx]
    if rgb is not None:
        rgb_s = rgb[idx].astype(np.int64)
    else:
        rgb_s = np.zeros((idx.shape[0], 3), dtype=np.int64)
    lines = [
        '{:.2f}, {:.2f}, {:.2f}, {}, {}, {} \n'.format(
            xyz_s[i, 0], xyz_s[i, 1], xyz_s[i, 2], rgb_s[i, 0], rgb_s[i, 1], rgb_s[i, 2])
        for i in range(xyz_s.shape[0])]
    return ''.join(lines)


def write_chunk_points_csv(
        path: str, tiles: typing.List[TileInfo], chunk_size: int = 50_000, points_per_chunk: int = 1,
        show_progress: bool = True):
    """
    Standalone version of the chunk-points sampling that build_heightmaps() can also do inline
    (via its chunk_points_csv_path argument, which avoids a second full-decode pass -- prefer
    that in generate_training_data(); this is kept for generating just the sparse sample on its
    own, e.g. for experimenting with different points_per_chunk values without re-binning
    heightmaps).

    IMPORTANT: despite chunk_size matching LASzip's compression chunk size, this does NOT try to
    save decompression work by seeking to each chunk boundary -- benchmarked on real tiles,
    laspy's LasReader.seek() to arbitrary chunk starts is far slower (~15x, likely per-seek
    decompressor-state setup cost) than just reading the whole tile once. So this just reads each
    tile fully and slices out the first `points_per_chunk` points of every `chunk_size`-sized run.
    """
    from source.base.fs import make_dir_for_file
    from tqdm import tqdm

    make_dir_for_file(path)
    iterator = tqdm(tiles, desc='Writing chunkPoints.csv') if show_progress else tiles
    with open(path, 'w') as f_out:
        for tile in iterator:
            xyz, rgb = _read_tile_points(tile.path)
            lines = _format_chunk_points_for_tile(xyz, rgb, chunk_size, points_per_chunk)
            if lines:
                f_out.write(lines)
            del xyz, rgb


def generate_training_data(
        pointcloud_dir: str, out_dir: str,
        num_query_points: int = 10_000, pixel_size: float = 10.0, heightmap_size: int = 64,
        chunk_size: int = 50_000, points_per_chunk: int = 1, seed: int = 0, show_progress: bool = True,
        num_workers: int = 1):
    """
    End-to-end port of main_create_training.cpp + heightmap_filter.mjs for one region/shape.

    num_workers > 1 parallelizes the binning pass across processes (see build_heightmaps).
    Memory scales with num_workers, not with dataset size -- each worker holds one decoded tile
    at a time. Pick a value based on available RAM alongside CPU count; e.g. os.cpu_count() is a
    reasonable starting point on a machine with plenty of memory to spare.
    """
    rng = np.random.default_rng(seed)

    files = list_las_files(pointcloud_dir)
    if not files:
        raise ValueError(f'No .las/.laz files found in {pointcloud_dir}')
    print(f'{pointcloud_dir}: {len(files)} tiles')

    tiles = gather_tile_info(files, show_progress=show_progress)
    total_points = sum(t.num_points for t in tiles)
    print(f'{len(tiles)} tiles with data, {total_points:,} points total')

    # resolves via targeted per-point seeks, not a full tile decode (see pick_query_points) --
    # cheap, and sorted by x afterward so a simple index-range split of the output (as
    # datasets/laz_minimal/*.txt already do) is spatially coherent, not spatially interleaved
    query_points_ms = pick_query_points(tiles, num_query_points, rng, show_progress=show_progress)

    # binning and chunk-points sampling folded into one pass over the tiles (each tile decoded
    # once), instead of two separate full-decode passes
    chunk_points_csv = os.path.join(out_dir, 'chunkPoints.csv')
    height, rgb, count = build_heightmaps(
        tiles, query_points_ms, heightmap_size=heightmap_size, pixel_size=pixel_size,
        show_progress=show_progress, chunk_points_csv_path=chunk_points_csv,
        chunk_size=chunk_size, points_per_chunk=points_per_chunk, num_workers=num_workers)
    print(f'wrote {chunk_points_csv}')

    heightmaps_bin = os.path.join(out_dir, 'heightmaps.bin')
    write_heightmaps_bin(heightmaps_bin, query_points_ms, height)
    print(f'wrote {heightmaps_bin}')

    if rgb is not None:
        for c in range(3):
            rgb_bin = os.path.join(out_dir, f'rgb_{c}.bin')
            write_heightmaps_bin(rgb_bin, query_points_ms, rgb[c])
            print(f'wrote {rgb_bin}')

    empty_fraction = float(np.mean(count.sum(axis=(1, 2)) == 0))
    if empty_fraction > 0:
        print(f'WARNING: {empty_fraction:.1%} of query points have zero points in every pixel '
             '(query point likely fell in a gap between tiles)')


def _naive_reference_binning(points_xyz, points_rgb, query_points_ms, heightmap_size, pixel_size):
    """Direct, unvectorized transliteration of main_create_training.cpp's per-point loop
    (using floor instead of the C++ truncation-towards-zero quirk, see module docstring).
    Used only to cross-check build_heightmaps()'s vectorized numpy version."""
    n = query_points_ms.shape[0]
    heightmap_world = heightmap_size * pixel_size
    half = heightmap_world / 2.0

    count = np.zeros((n, heightmap_size, heightmap_size), dtype=np.int64)
    z_sum = np.zeros((n, heightmap_size, heightmap_size), dtype=np.float64)
    rgb_sum = np.zeros((3, n, heightmap_size, heightmap_size), dtype=np.float64)

    for qi in range(n):
        world_min = query_points_ms[qi, :2] - half
        for pi in range(points_xyz.shape[0]):
            u = (points_xyz[pi, 0] - world_min[0]) / heightmap_world
            v = (points_xyz[pi, 1] - world_min[1]) / heightmap_world
            px = int(np.floor(heightmap_size * u))
            py = int(np.floor(heightmap_size * v))
            if px < 0 or px > heightmap_size - 1 or py < 0 or py > heightmap_size - 1:
                continue
            count[qi, py, px] += 1
            z_sum[qi, py, px] += points_xyz[pi, 2]
            if points_rgb is not None:
                for c in range(3):
                    rgb_sum[c, qi, py, px] += points_rgb[pi, c]

    height = np.where(count > 0, z_sum / np.maximum(count, 1), np.nan)
    rgb = np.where(count[None] > 0, rgb_sum / np.maximum(count, 1)[None], np.nan) if points_rgb is not None else None
    return height, rgb, count


def _unit_test_resort_heightmaps_bin_by_query_x():
    import tempfile

    rng = np.random.default_rng(4)
    n, s = 25, 4
    query_points_ms = rng.uniform(-100.0, 100.0, size=(n, 3))
    height = rng.uniform(0.0, 50.0, size=(n, s, s)).astype(np.float32)
    rgb = [rng.uniform(0.0, 1.0, size=(n, s, s)).astype(np.float32) for _ in range(3)]

    expected_order = np.argsort(query_points_ms[:, 0], kind='stable')

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_heightmaps_bin(os.path.join(tmp_dir, 'heightmaps.bin'), query_points_ms, height)
        for c in range(3):
            write_heightmaps_bin(os.path.join(tmp_dir, f'rgb_{c}.bin'), query_points_ms, rgb[c])

        resort_heightmaps_bin_by_query_x(tmp_dir, heightmap_size=s)

        dt = np.dtype('3f8, ({0},{0})f4'.format(s))
        hm_data = np.fromfile(os.path.join(tmp_dir, 'heightmaps.bin'), dtype=dt)
        assert np.allclose(hm_data['f0'], query_points_ms[expected_order])
        assert np.allclose(hm_data['f1'], height[expected_order])
        for c in range(3):
            rgb_data = np.fromfile(os.path.join(tmp_dir, f'rgb_{c}.bin'), dtype=dt)
            assert np.allclose(rgb_data['f1'], rgb[c][expected_order])

        # idempotent: resorting an already-sorted file is a no-op
        resort_heightmaps_bin_by_query_x(tmp_dir, heightmap_size=s)
        hm_data_2 = np.fromfile(os.path.join(tmp_dir, 'heightmaps.bin'), dtype=dt)
        assert np.array_equal(hm_data['f0'], hm_data_2['f0'])
    print('_unit_test_resort_heightmaps_bin_by_query_x: OK')


def _write_temp_las_tile(path: str, xyz: np.ndarray, rgb: np.ndarray):
    import laspy

    header = laspy.LasHeader(point_format=3)
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.red = rgb[:, 0].astype(np.uint16)
    las.green = rgb[:, 1].astype(np.uint16)
    las.blue = rgb[:, 2].astype(np.uint16)
    las.write(path)


def _unit_test_pick_query_points():
    """Cross-checks pick_query_points()'s seek-based xyz resolution against directly indexing
    the fully-decoded points, and checks the result is sorted by x."""
    import tempfile

    rng = np.random.default_rng(1)
    num_points = 400
    xyz = rng.uniform(low=0.0, high=1000.0, size=(num_points, 3))
    rgb = rng.integers(low=0, high=256, size=(num_points, 3)).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        las_path = f'{tmp_dir}/tile.las'
        _write_temp_las_tile(las_path, xyz, rgb)

        tiles = gather_tile_info([las_path], show_progress=False)
        num_query_points = 37
        result = pick_query_points(tiles, num_query_points, np.random.default_rng(2), show_progress=False)

    assert result.shape == (num_query_points, 3)
    assert np.all(np.diff(result[:, 0]) >= 0), 'result is not sorted by x'
    # every resolved point must exactly match one of the original tile points (seek-based
    # resolution must not silently return garbage or a different point)
    for row in result:
        dists = np.sum((xyz - row[None, :]) ** 2, axis=1)
        assert dists.min() < 1e-6, 'resolved query point does not match any original tile point'
    print(f'_unit_test_pick_query_points: OK ({num_query_points} query points, {num_points} points)')


def _unit_test_build_heightmaps_chunk_points_inline():
    """Checks that build_heightmaps()'s inline chunk_points_csv_path writing matches the
    standalone write_chunk_points_csv() writing the same tiles."""
    import tempfile
    import filecmp

    rng = np.random.default_rng(3)
    num_points = 300
    xyz = rng.uniform(low=-50.0, high=50.0, size=(num_points, 3))
    rgb = rng.integers(low=0, high=256, size=(num_points, 3)).astype(np.uint8)
    query_points_ms = np.array([[0.0, 0.0, 0.0]])

    with tempfile.TemporaryDirectory() as tmp_dir:
        las_path = f'{tmp_dir}/tile.las'
        _write_temp_las_tile(las_path, xyz, rgb)
        tiles = gather_tile_info([las_path], show_progress=False)

        inline_csv = f'{tmp_dir}/inline.csv'
        standalone_csv = f'{tmp_dir}/standalone.csv'
        build_heightmaps(
            tiles, query_points_ms, heightmap_size=8, pixel_size=10.0, show_progress=False,
            chunk_points_csv_path=inline_csv, chunk_size=37, points_per_chunk=3)
        write_chunk_points_csv(standalone_csv, tiles, chunk_size=37, points_per_chunk=3, show_progress=False)

        assert filecmp.cmp(inline_csv, standalone_csv, shallow=False), (
            'inline chunk-points output differs from the standalone writer')
    print('_unit_test_build_heightmaps_chunk_points_inline: OK')


def _unit_test_build_heightmaps():
    """Cross-checks the vectorized build_heightmaps() against a naive per-point-loop reference
    on synthetic data, using the real code path (a temp dir with one tiny synthetic .las tile)."""
    import tempfile

    rng = np.random.default_rng(0)
    heightmap_size = 8
    pixel_size = 10.0

    num_points = 500
    xyz = rng.uniform(low=-20.0, high=100.0, size=(num_points, 3))
    xyz[:, 2] = rng.uniform(low=0.0, high=50.0, size=num_points)
    rgb = rng.integers(low=0, high=256, size=(num_points, 3)).astype(np.uint8)

    query_points_ms = np.array([[20.0, 20.0, 10.0], [0.0, 0.0, 5.0]])
    # add enough extra query points to exercise batching (default max_windows_per_batch=16) with
    # a mix of windows that do and don't intersect the point cloud
    extra_xy = rng.uniform(low=-40.0, high=120.0, size=(22, 2))
    extra_z = rng.uniform(low=0.0, high=50.0, size=(22, 1))
    query_points_ms = np.concatenate([query_points_ms, np.concatenate([extra_xy, extra_z], axis=1)], axis=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        las_path = f'{tmp_dir}/tile.las'
        _write_temp_las_tile(las_path, xyz, rgb)

        tiles = gather_tile_info([las_path], show_progress=False)
        height, rgb_out, count = build_heightmaps(
            tiles, query_points_ms, heightmap_size=heightmap_size, pixel_size=pixel_size, show_progress=False,
            max_windows_per_batch=5)  # smaller than num affected query points, to exercise batch boundaries

    height_ref, rgb_ref, count_ref = _naive_reference_binning(
        xyz, rgb.astype(np.float64), query_points_ms, heightmap_size, pixel_size)

    assert np.array_equal(count, count_ref), 'pixel occupancy counts differ from the reference loop'
    assert np.allclose(height, height_ref, equal_nan=True, atol=1e-3), 'mean height differs from the reference loop'
    assert np.allclose(rgb_out, rgb_ref, equal_nan=True, atol=1.0), 'mean RGB differs from the reference loop'
    print('_unit_test_build_heightmaps: OK ({} query points, {} points, {} occupied pixels)'.format(
        query_points_ms.shape[0], num_points, int((count > 0).sum())))


def _unit_test_build_heightmaps_parallel():
    """Checks that build_heightmaps(num_workers>1) produces identical output to the serial path
    (num_workers=1), including the inline chunk-points CSV, across multiple tiles."""
    import tempfile

    rng = np.random.default_rng(5)
    heightmap_size = 8
    pixel_size = 10.0
    num_tiles = 4

    query_points_ms = np.stack([
        rng.uniform(-30.0, 130.0, size=20),
        rng.uniform(-30.0, 130.0, size=20),
        rng.uniform(0.0, 50.0, size=20),
    ], axis=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tile_paths = []
        for t in range(num_tiles):
            num_points = 150
            xyz = rng.uniform(low=t * 25.0, high=t * 25.0 + 60.0, size=(num_points, 3))
            xyz[:, 2] = rng.uniform(low=0.0, high=50.0, size=num_points)
            rgb = rng.integers(low=0, high=256, size=(num_points, 3)).astype(np.uint8)
            path = f'{tmp_dir}/tile_{t}.las'
            _write_temp_las_tile(path, xyz, rgb)
            tile_paths.append(path)

        tiles = gather_tile_info(tile_paths, show_progress=False)

        csv_serial = f'{tmp_dir}/serial.csv'
        height_s, rgb_s, count_s = build_heightmaps(
            tiles, query_points_ms, heightmap_size=heightmap_size, pixel_size=pixel_size, show_progress=False,
            chunk_points_csv_path=csv_serial, chunk_size=37, points_per_chunk=2, num_workers=1)

        csv_parallel = f'{tmp_dir}/parallel.csv'
        height_p, rgb_p, count_p = build_heightmaps(
            tiles, query_points_ms, heightmap_size=heightmap_size, pixel_size=pixel_size, show_progress=False,
            chunk_points_csv_path=csv_parallel, chunk_size=37, points_per_chunk=2, num_workers=3)

        # imap_unordered means tile order (and thus chunk-points line order) may differ -- compare as sorted lines
        with open(csv_serial) as f:
            lines_s = sorted(f.readlines())
        with open(csv_parallel) as f:
            lines_p = sorted(f.readlines())

    assert np.array_equal(count_s, count_p), 'parallel counts differ from serial'
    assert np.allclose(height_s, height_p, equal_nan=True), 'parallel heights differ from serial'
    assert np.allclose(rgb_s, rgb_p, equal_nan=True), 'parallel rgb differs from serial'
    assert lines_s == lines_p, 'parallel chunk-points CSV content differs from serial (ignoring order)'
    print('_unit_test_build_heightmaps_parallel: OK ({} tiles, {} query points, {} occupied)'.format(
        num_tiles, query_points_ms.shape[0], int((count_s > 0).sum())))


if __name__ == '__main__':
    _unit_test_build_heightmaps()
    _unit_test_pick_query_points()
    _unit_test_build_heightmaps_chunk_points_inline()
    _unit_test_resort_heightmaps_bin_by_query_x()
    _unit_test_build_heightmaps_parallel()
