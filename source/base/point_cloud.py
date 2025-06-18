import typing

import numpy as np

from source.base import fs


def load_xyz(file_path):
    data = np.loadtxt(file_path).astype('float32')
    nan_lines = np.isnan(data).any(axis=1)
    num_nan_lines = np.sum(nan_lines)
    if num_nan_lines > 0:
        data = data[~nan_lines]  # filter rows with nan values
        print('Ignored {} points containing NaN coordinates in point cloud {}'.format(num_nan_lines, file_path))
    return data


def write_ply(file_path: str, points: np.ndarray, normals=None, colors=None):
    """
    Write point cloud file as .ply.
    :param file_path:
    :param points:
    :param normals:
    :param colors:
    :return: None
    """

    import trimesh

    assert (file_path.endswith('.ply'))

    def sanitize_inputs(arr: np.ndarray):
        if arr is None:
            return arr

        # should be array
        arr = np.asarray(arr)

        # should be 2 dims
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        # convert 2d points to 3d
        if arr.shape[1] == 2:
            arr_2p5d = np.zeros((arr.shape[0], 3))
            arr_2p5d[:, :2] = arr
            arr_2p5d[:, 2] = 0.0
            arr = arr_2p5d

        # should be (n, dims)
        if arr.shape[0] == 3 and arr.shape[1] != 3:
            arr = arr.transpose([1, 0])

        if arr.shape[-1] != 3:
            raise ValueError('Unexpected shape for point data: {}'.format(arr.shape))

        return arr

    points = sanitize_inputs(points)
    colors = sanitize_inputs(colors)
    normals = sanitize_inputs(normals)

    mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors, vertex_normals=normals)
    fs.make_dir_for_file(file_path)
    mesh.export(file_path)


def pts_to_img(
        pts_ps_xy: np.ndarray, pts_data: np.ndarray, resolution: int,
        method: typing.Literal[
            'rasterize',
            'rast_nearest_min', 'rast_nearest_max', 'rast_nearest_mean',
            'rast_linear_min', 'rast_linear_max', 'rast_linear_mean',
            'rast_rbflinear_min', 'rast_rbflinear_max', 'rast_rbflinear_mean',
            'nearest', 'linear', 'cubic',
            'hqsplat', 'knngauss'],
        border_pixels=16, context_radius_factor=1.5):

    if method == 'rasterize':
        return rasterize_pts(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, context_radius_factor=context_radius_factor,
            resolution=resolution, border_pixels=border_pixels, interp_method='rasterize')
    elif method.startswith('rast_'):
        _, interp_method, same_pixel_method = method.split('_')
        if interp_method == 'pyramid':
            return rast_pyramid(
                pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution,
                context_radius_factor=context_radius_factor, border_pixels=border_pixels)
        else:
            return rasterize_pts(
                pts_ps_xy=pts_ps_xy, pts_data=pts_data, border_pixels=border_pixels, context_radius_factor=context_radius_factor,
                resolution=resolution, interp_method=interp_method, same_pixel_method=same_pixel_method)
    elif method == 'nearest':
        return interpolate_patch(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method='nearest')
    elif method == 'linear':
        return interpolate_patch(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method='linear')
    elif method == 'cubic':
        return interpolate_patch(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method='cubic')
    elif method == 'knngauss':
        return knn_gauss(pts_ps_xy=pts_ps_xy, pts_data=pts_data, res=resolution)
    else:
        raise ValueError(f'Unknown method: {method}')


def pts_to_img_cached(pts_ps_xy: np.ndarray, pts_data: np.ndarray, resolution: int, method: str, cache_dir: str):
    import os
    from source.base import fs

    # if data is only NaNs, return directly
    if np.isnan(pts_data).all():  # RGB is often NaN
        if len(pts_data.shape) == 1:
            return np.full((resolution, resolution), np.nan)
        else:
            return np.full((pts_data.shape[1], resolution, resolution), np.nan)

    input_hash = fs.md5(pts_ps_xy.tobytes()) + fs.md5(pts_data.tobytes()) + resolution + fs.str_to_consistent_hash(method)
    cache_file = os.path.join(cache_dir, '{}.npy'.format(input_hash))

    # killing the process may leave empty cache files
    # if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0 and method != 'rasterize':  # rasterize ist fast
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        return np.load(cache_file)

    fs.make_dir_for_file(cache_file)
    img = pts_to_img(pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=resolution, method=method)
    np.save(cache_file, img)
    return img


def rast_pyramid(
        pts_ps_xy: np.ndarray, pts_data: np.ndarray, resolution: int,
        context_radius_factor=1.5, border_pixels=16, pyramid_levels=5):

    pyramid_resolutions = [resolution // (2 ** i) for i in range(pyramid_levels)]
    pyramid_img_min = []
    pyramid_img_max = []
    pyramid_img_mean = []
    for res in pyramid_resolutions:
        img_min = rasterize_pts(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=res, context_radius_factor=context_radius_factor,
            border_pixels=border_pixels, interp_method='rasterize', same_pixel_method='min')
        pyramid_img_min.append(img_min)
        img_max = rasterize_pts(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=res, context_radius_factor=context_radius_factor,
            border_pixels=border_pixels, interp_method='rasterize', same_pixel_method='max')
        pyramid_img_max.append(img_max)
        img_mean = rasterize_pts(
            pts_ps_xy=pts_ps_xy, pts_data=pts_data, resolution=res, context_radius_factor=context_radius_factor,
            border_pixels=border_pixels, interp_method='rasterize', same_pixel_method='mean')
        pyramid_img_mean.append(img_mean)

    for i in range(len(pyramid_resolutions)):
        pyramid_img_min[i] = np.repeat(pyramid_img_min[i], 2 ** i, axis=0)
        pyramid_img_min[i] = np.repeat(pyramid_img_min[i], 2 ** i, axis=1)
        pyramid_img_max[i] = np.repeat(pyramid_img_max[i], 2 ** i, axis=0)
        pyramid_img_max[i] = np.repeat(pyramid_img_max[i], 2 ** i, axis=1)
        pyramid_img_mean[i] = np.repeat(pyramid_img_mean[i], 2 ** i, axis=0)
        pyramid_img_mean[i] = np.repeat(pyramid_img_mean[i], 2 ** i, axis=1)

    pyramid_img_repeated = np.stack(pyramid_img_min + pyramid_img_max + pyramid_img_mean, axis=0)

    pyramid_img_mean_nan = np.isnan(pyramid_img_mean)  # per level, same mask for min/max/mean
    # pyramid_img_mean_nan = np.isnan(pyramid_img_repeated)  # per level repeated for min/max/mean
    pyramid_img_repeated[np.isnan(pyramid_img_repeated)] = 0.0
    pyramid_img_repeated = np.concatenate([pyramid_img_repeated, pyramid_img_mean_nan.astype(np.float32)], axis=0)

    return pyramid_img_repeated

def rasterize_pts(
        pts_ps_xy: np.ndarray, pts_data: np.ndarray, resolution: int,
        context_radius_factor=1.5, border_pixels=16,
        interp_method: typing.Literal['rasterize', 'nearest', 'linear', 'rbflinear', 'hqsplat'] = 'nearest',
        same_pixel_method: typing.Literal['first', 'min', 'max', 'mean'] = 'mean'):
    """
    Rasterize points to heightmap
    Note: points outside [-1, 1] cause errors
    pts: [k, 3]
    resolution: int
    fill_value: None|float
    return: [resolution, resolution]
    """
    from source.base.math import normalize_data
    from source.base.img import slice_img_center, rasterized_to_nearest, rasterized_to_linear, rasterized_to_hqsplat
    from scipy.sparse import coo_matrix

    # avoid ugly artifacts at the border
    # scale by border_pixels, normalize to larger resolution, slice center
    resolution_border = resolution + 2 * border_pixels
    scaling_factor = resolution / resolution_border
    pts_ps_xy_norm = normalize_data(arr=pts_ps_xy * context_radius_factor * np.sqrt(2) * scaling_factor,
                                    in_min=-1.0, in_max=1.0,
                                    out_min=0, out_max=resolution_border-1, clip=True)

    pts_coo = np.round(pts_ps_xy_norm).astype(int)

    if same_pixel_method == 'first':
        pts_coo_unique, unique_indices = np.unique(pts_coo, axis=0, return_index=True)
        pts_data_unique = pts_data[unique_indices]
    elif same_pixel_method in ['min', 'max', 'mean']:
        # sort and group by occupied pixel
        sort_ids = np.lexsort(np.flipud(pts_coo.T), axis=0)
        coo_ids_sorted = pts_coo[sort_ids]
        _, unique_pts_pixel_ids = np.unique(coo_ids_sorted, axis=0, return_index=True)
        sort_ids_groups = np.split(sort_ids, unique_pts_pixel_ids)
        sort_ids_groups = [group for group in sort_ids_groups if group.shape[0] > 0]

        if same_pixel_method == 'min':
            pts_data_unique = np.asarray(
                [pts_data[group].min(axis=0) for group in sort_ids_groups])
        elif same_pixel_method == 'max':
            pts_data_unique = np.asarray(
                [pts_data[group].max(axis=0) for group in sort_ids_groups])
        elif same_pixel_method == 'mean':
            pts_data_unique = np.asarray(
                [pts_data[group].mean(axis=0) for group in sort_ids_groups])

        # first point of each group
        pts_coo_unique = np.stack([pts_coo[group[0]] for group in sort_ids_groups], axis=0)
        pass
    else:
        raise ValueError(f'Unknown same_pixel_method: {same_pixel_method}')

    pts_coo_unique = (pts_coo_unique[..., 0], pts_coo_unique[..., 1])

    mat_shape = (resolution_border, resolution_border)
    has_data_channels = len(pts_data_unique.shape) > 1
    if not has_data_channels:
        pts_data_unique = np.expand_dims(pts_data_unique, axis=1)
    data_channels = pts_data_unique.shape[1]
    imgs_per_channel = []
    for i in range(data_channels):
        sparse_mat = coo_matrix((pts_data_unique[:, i], pts_coo_unique), shape=mat_shape)

        hm = sparse_mat.A
        hm = hm.T
        hm[hm == 0.0] = np.nan  # mark zeros as unknown

        if interp_method == 'nearest':
            hm, dists = rasterized_to_nearest(hm)
        elif interp_method == 'linear':
            hm, values_sum, weights = rasterized_to_linear(hm)
        elif interp_method == 'rasterize':
            pass
        elif interp_method == 'rbflinear':
            from scipy.interpolate import Rbf
            rbf = Rbf(*pts_coo_unique, pts_data_unique[:, i], function='linear')
            x, y = np.mgrid[0:resolution_border, 0:resolution_border]
            hm = rbf(x, y).T
        elif interp_method == 'hqsplat':
            hm, values_sum, weights = rasterized_to_hqsplat(hm)
        else:
            raise ValueError(f'Unknown interp_method: {interp_method}')
        imgs_per_channel.append(hm)

    if data_channels == 1:
        hm = imgs_per_channel[0]
    else:
        hm = np.stack(imgs_per_channel, axis=0)

    hm = slice_img_center(hm, resolution_border, resolution)
    return hm

def get_magic_scaling_factor():
    # might be related to pixel center vs corner
    # TODO: find out why this is necessary, seems to come from the dataset generation
    # magic_scaling_factor = 102/96  # RMSE 34.9
    magic_scaling_factor = 103/96  # RMSE 34.26
    # magic_scaling_factor = 104/96  # RMSE 35.35
    return magic_scaling_factor


def interpolate_patch(pts_ps_xy, pts_data: np.ndarray, resolution, method='linear'):
    from scipy.interpolate import griddata

    if pts_ps_xy.shape[0] < 3:
        print('WARNING: At least 3 points are required for triangulation')
        return np.zeros((resolution, resolution))

    pts_ps_xy = pts_ps_xy * get_magic_scaling_factor()

    # revert patch normalization
    steps = complex(0.0, resolution)  # special indexing for mgrid
    hm_size_ps = 0.5  # no extra padding

    grid_x, grid_y = np.mgrid[-hm_size_ps:hm_size_ps:steps, -hm_size_ps:hm_size_ps:steps]
    grid_z_ps_linear = griddata(pts_ps_xy, pts_data, (grid_x, grid_y), method=method)

    # interpolation outside convex hull is nan -> fill with nearest
    unknown = np.isnan(grid_z_ps_linear)
    if unknown.any():
        grid_z_ps_nearest = griddata(pts_ps_xy, pts_data, (grid_x, grid_y), method='nearest')
        grid_z_ps_linear[unknown] = grid_z_ps_nearest[unknown]

    # fix ij indexing of mgrid
    grid_z_ps_linear = grid_z_ps_linear.T

    return grid_z_ps_linear


def knn_gauss(pts_ps_xy: np.ndarray, pts_data: np.ndarray, res: int, padding: int = 16, k=20):
    # build kdtree for patch
    # get k NNs for each pixel
    # get weights with Gaussian
    # get weighted average
    from scipy.spatial import KDTree
    from scipy.stats import norm
    from source.base.img import slice_img_center

    # turn data into vectors if it is not already
    single_channel = len(pts_data.shape) == 1
    if single_channel:
        pts_data = np.expand_dims(pts_data, axis=1)

    res_padded = res + 2 * padding
    scaling_factor = res_padded / res / get_magic_scaling_factor()
    pts_ps_xy = pts_ps_xy * scaling_factor
    kdtree = KDTree(pts_ps_xy)
    grid_x, grid_y = np.mgrid[-1:1:complex(0.0, res_padded), -1:1:complex(0.0, res_padded)]
    grid_xy = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    nn_dists, nn_indices = kdtree.query(grid_xy, k=k, workers=8)
    nn_weights = norm.pdf(nn_dists, loc=0, scale=0.5)
    nn_weights /= nn_weights.sum(axis=1, keepdims=True)
    # https://www.wolframalpha.com/input?i=normal+distribution+pdf+scale%3D1
    # (norm.pdf(nn_dists, loc=0, scale=0.5) / norm.pdf(nn_dists, loc=0, scale=0.5).sum(axis=1, keepdims=True))[4200]
    grid_data_padded = np.sum(pts_data[nn_indices] * nn_weights[:, :, np.newaxis], axis=1)

    grid_data_padded = grid_data_padded.reshape(res_padded, res_padded, pts_data.shape[1])  # flat to grid
    grid_data_padded = grid_data_padded.transpose((1, 0, 2))  # fix ij indexing of mgrid
    grid_data_padded = grid_data_padded.transpose((2, 0, 1))  # move channels to front

    grid_data = slice_img_center(grid_data_padded, res_padded, res)

    # unsqueeze if only one channel
    if single_channel:
        grid_data = grid_data[0]
    return grid_data


def las2dem(las_in: str, img_out: str, extra_args: str = '',
            lastools_dir: str = 'C:\\Program Files\\LAStools\\bin\\') -> None:
    # Overview: https://rapidlasso.de/generating-spike-free-digital-surface-models-from-lidar/
    # Params: https://downloads.rapidlasso.de/readme/las2dem_README.md
    import os
    from subprocess import CompletedProcess
    import subprocess
    from source.base.fs import make_dir_for_file

    file_in_abs = os.path.abspath(las_in)
    file_out_abs = os.path.abspath(img_out)
    # verbose = '-v' if verbose else ''
    make_dir_for_file(file_out_abs)

    cmd = f'"{lastools_dir}las2dem.exe" -i "{file_in_abs}" -o "{file_out_abs}" ' + extra_args
    # print(cmd)
    proc: CompletedProcess = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(proc)
        print(proc.stdout)
        print(proc.stderr)
        raise ValueError('las2dem failed with return code: {}'.format(proc.returncode))


def las2dem_np(las_in: str, bil_file: str, png_file: str, extra_args: str = '',
               lastools_dir: str = 'C:\\Program Files\\LAStools\\bin\\') -> tuple[np.ndarray, np.ndarray]:
    from source.base.bil_parse import BilParser
    import imageio.v3 as iio

    hm_png = bil_file.replace('.bil', '.png')

    las2dem(las_in, hm_png, ' -elevation ' + extra_args, lastools_dir)
    las2dem(las_in, bil_file, ' -nbits 16 ' + extra_args, lastools_dir)
    las2dem(las_in, png_file, ' -rgb ' + extra_args, lastools_dir)

    # BIL, ASC, IMG, DTM, and XYZ, recommended for elevation
    bil_file_header = bil_file.replace('.bil', '.hdr')
    hm_bil = BilParser(bil_file_header)
    hm = hm_bil.values
    rgb = iio.imread(png_file)
    return hm, rgb


def np_to_las(pts_local_ms: np.ndarray, pts_local_rgb: np.ndarray, las_file: str):
    import laspy
    from source.base.fs import make_dir_for_file

    red = np.array(pts_local_rgb[:, 0]) * 255  # get rid of tracked array
    green = np.array(pts_local_rgb[:, 1]) * 255
    blue = np.array(pts_local_rgb[:, 2]) * 255

    las = laspy.create(file_version="1.2", point_format=2)
    las.xyz = pts_local_ms
    las.red = red
    las.green = green
    las.blue = blue
    make_dir_for_file(las_file)
    las.write(las_file)
    write_ply(las_file.replace('.las', '.ply'), pts_local_ms, colors=pts_local_rgb)


def ply_to_las(pts_in: str, pts_out: str) -> None:
    import trimesh
    pts: trimesh.PointCloud = trimesh.load(pts_in)
    np_to_las(pts.vertices, pts.colors, pts_out)



def _unit_test_rasterize():
    from source.base.profiling import get_duration

    # pts_xyz = np.array([
    #     [-0.25, -0.25, 0.3],
    #     [0.4, -0.4, 1.0],
    #     [0.4, -0.4, 0.8],
    #     [-0.25, -0.25, 0.2],
    #     [0.5, 0.5, 0.5],
    #     [-0.3, 0.3, 0.6],])
    pts_xyz = np.random.rand(1000, 3) - 0.5
    # resolution = 10
    resolution = 96

    pts = pts_xyz[:, :2]
    data = pts_xyz[:, 2]
    methods = [
        'rasterize',
        'rast_nearest_min', 'rast_nearest_max', 'rast_nearest_mean',
        'rast_linear_min', 'rast_linear_max', 'rast_linear_mean',
        # 'rast_rbflinear_min', 'rast_rbflinear_max', 'rast_rbflinear_mean',
        'nearest', 'linear', 'cubic'
    ]
    hms = []
    durations = []
    for m in methods:
        params = {'pts_ps_xy': pts, 'pts_data': data, 'border_pixels': 2, 'resolution': resolution, 'method': m}
        dur, img = get_duration(func=pts_to_img, params=params, repeat=10)
        # img = pts_to_img(pts_ps_xy=pts, pts_data=data, resolution=resolution, border_pixels=2, method=m)
        hms.append(img)
        durations.append(dur)

    # same colormap for all
    from matplotlib import pyplot as plt
    num_plots = len(methods) + 1
    aspect_ratio = 4 / 3
    num_rows = int(np.sqrt(num_plots / aspect_ratio))
    num_cols = int(np.ceil(num_plots / num_rows))
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
    ax = ax.flatten()
    for i, hm in enumerate(hms):
        im_last = ax[i].imshow(hm, vmin=0.0, vmax=1.0)
        title_method_duration = '{}\n{:.3f}ms'.format(methods[i], durations[i] * 1000.0)
        ax[i].set_title(title_method_duration)
    fig.colorbar(im_last, cax=ax[-1])
    plt.show()


def _unit_test_ply_to_las():
    pts_ply = 'datasets/laz_minimal/bins/ca_13/chunkPoints.ply'
    bil_file = pts_ply.replace('.ply', '.bil')
    png_file = pts_ply.replace('.ply', '.png')
    pts_out = pts_ply.replace('.ply', '.las')
    verbose = '-v'
    step_m = 10.0
    kill_m = 1000
    extra_args = f'{verbose} -step {step_m} -kill {kill_m}'
    ply_to_las(pts_ply, pts_out)
    hm, rgb = las2dem_np(pts_out, bil_file, png_file, extra_args)

    print(hm[:200, :200])
    print(rgb[:200, :200])

    return 0


if __name__ == '__main__':
    _unit_test_rasterize()
    _unit_test_ply_to_las()
