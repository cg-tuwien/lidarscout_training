import typing

import numpy as np
import trimesh


def texel_to_pts(ixy: np.ndarray, height: np.ndarray, query_pts: np.ndarray, hm_size: int, pixel_size: float = 10.0):
    pts_xy_ms_relative = (ixy - (hm_size / 2)) * pixel_size
    pts_xy_ms = pts_xy_ms_relative + query_pts[np.newaxis, :2]
    pts_ms = np.concatenate((pts_xy_ms, height[..., np.newaxis]), axis=1)
    return pts_ms


def hm_to_normals(hm: np.ndarray, pixel_size: float = 10.0):
    hm_padded = np.pad(hm, 1, mode='constant', constant_values=np.nan)

    # continue padding with same gradient
    hm_padded[0, :] = hm_padded[1, :] + (hm_padded[1, :] - hm_padded[2, :])
    hm_padded[-1, :] = hm_padded[-2, :] + (hm_padded[-2, :] - hm_padded[-3, :])
    hm_padded[:, 0] = hm_padded[:, 1] + (hm_padded[:, 1] - hm_padded[:, 2])
    hm_padded[:, -1] = hm_padded[:, -2] + (hm_padded[:, -2] - hm_padded[:, -3])

    hm_top = hm_padded[:-2, 1:-1]
    hm_bottom = hm_padded[2:, 1:-1]
    hm_right = hm_padded[1:-1, :-2]
    hm_left = hm_padded[1:-1, 2:]

    normal_x = 2.0 * (hm_right - hm_left) / pixel_size
    normal_y = 2.0 * (hm_top - hm_bottom) / pixel_size
    normal_z = np.full_like(normal_x, 4.0)
    normals = np.stack((normal_x, normal_y, normal_z), axis=2)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    return normals


def hm_to_pts(hm: np.ndarray, query_pts: np.ndarray, pixel_size: float = 10.0):
    """
    heightmap to points with normals
    :param hm: np.ndarray[hm_size, hm_size]
    :param query_pts:
    :param pixel_size:
    :return:
    """
    hm_size = hm.shape[-1]
    ixy = np.fliplr(np.indices((hm_size, hm_size)).reshape(2, -1).T)  # must match order in bin file
    pts_ms = texel_to_pts(ixy=ixy, height=hm.flatten(),
                          query_pts=query_pts, hm_size=hm_size, pixel_size=pixel_size)
    pts_normals = hm_to_normals(hm=hm, pixel_size=pixel_size)
    pts_normals_flat = pts_normals.reshape(-1, 3)
    return pts_ms, pts_normals_flat


def normalize_data(arr: np.ndarray, in_max: float, in_min: float, out_max=1.0, out_min=-1.0, clip=False):

    arr = arr.copy()
    in_range = in_max - in_min
    out_range = out_max - out_min

    if in_range == 0.0 or out_range == 0.0:
        print('Warning: normalization would result in NaN, kept raw values')
        return arr - in_max

    # scale so that in_max=1.0 and in_min=0.0
    arr -= in_min
    arr /= in_range

    # scale to out_max..out_min
    arr *= out_range
    arr += out_min

    if clip:
        arr = np.clip(arr, out_min, out_max)

    return arr
