"""
Generate LidarScout training data (heightmaps.bin, rgb_*.bin, chunkPoints.csv) from a folder of
LAS/LAZ tiles, without needing the separate simlod C++ build.

Python port of:
  - .../simlod/src/main_create_training.cpp
  - .../simlod/tools/heightmap_filter.mjs
See source/base/dataset_generation.py for the implementation and a list of deliberate
deviations from the original C++ tool.

Usage:
    python create_training_data.py --pointcloud_dir "C:/path/to/tiles" --out_dir "datasets/laz_minimal/bins/my_shape"

Memory: tiles are processed one at a time, so peak memory is bounded by the largest single
tile, not the size of the whole dataset -- safe for tens to hundreds of GB of input tiles.
"""
import argparse

from source.base.dataset_generation import generate_training_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pointcloud_dir', type=str, required=True, help='directory containing .las/.laz tiles')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory, e.g. datasets/laz_minimal/bins/<shape_name>')
    parser.add_argument('--num_query_points', type=int, default=10_000)
    parser.add_argument('--pixel_size', type=float, default=10.0, help='meters per heightmap pixel')
    parser.add_argument('--heightmap_size', type=int, default=64, help='heightmap resolution in pixels')
    parser.add_argument('--chunk_size', type=int, default=50_000,
                        help='points per LAZ chunk (matches LASzip default); chunkPoints.csv gets '
                             '--points_per_chunk points from the start of every chunk')
    parser.add_argument('--points_per_chunk', type=int, default=1,
                        help='1 reproduces the original heightmap_filter.mjs sparse sample; >1 is '
                             'the denser "LOD-less LOD" sample (see plan/notes)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true', help='disable progress bars')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='parallelize the binning pass across this many processes. Memory '
                             'scales with num_workers (each worker holds one decoded tile at a '
                             'time), not with dataset size, so on a machine with many cores and '
                             'plenty of RAM to spare, os.cpu_count() is a reasonable value.')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_training_data(
        pointcloud_dir=args.pointcloud_dir,
        out_dir=args.out_dir,
        num_query_points=args.num_query_points,
        pixel_size=args.pixel_size,
        heightmap_size=args.heightmap_size,
        chunk_size=args.chunk_size,
        points_per_chunk=args.points_per_chunk,
        seed=args.seed,
        show_progress=not args.quiet,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
