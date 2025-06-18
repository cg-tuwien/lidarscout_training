import numpy as np

from source.base import fs


def slice_img_center(img: np.ndarray, res_in: int, res_out: int, channels_first=True) -> np.ndarray:
    if res_in == res_out:
        return img
    elif res_in < res_out:
        raise ValueError('Upsampling not supported')
    else:
        diff = (res_in - res_out) // 2
        if channels_first:
            return img[..., diff:diff + res_out, diff:diff + res_out]
        else:
            return img[diff:diff + res_out, diff:diff + res_out]


def save_img(arr: np.ndarray, file: str, norm_min=0.0, norm_max=1.0):
    """
    Save an image to a file.
    :param arr: image data, [H, W] or [H, W, 3] in range [0, 1]
    :param file: output file
    :param norm_min: minimum value for normalization
    :param norm_max: maximum value for normalization
    """
    from source.base.math import normalize_data
    from imageio import imwrite

    arr_norm = normalize_data(arr, clip=True,
                              in_min=norm_min, in_max=norm_max,
                              out_min=0.0, out_max=255.0)
    arr_norm = arr_norm.astype(np.uint8)

    # repeat to imitate RGB image
    if len(arr_norm.shape) == 2:
        arr_norm = np.repeat(arr_norm[:, :, np.newaxis], 3, axis=2)

    fs.make_dir_for_file(file)
    imwrite(file, arr_norm)


def rasterized_to_nearest(img: np.ndarray) -> (np.ndarray, np.ndarray):
    from scipy.ndimage import distance_transform_edt
    unknown = np.isnan(img)
    distances, ids = distance_transform_edt(unknown, return_distances=True, return_indices=True)
    img[unknown] = img[tuple(ids[:, unknown])]
    return img, distances


def rasterized_to_linear(img: np.ndarray):

    if np.all(np.isnan(img)):  # happens e.g. when no RGB is available
        return img, np.zeros_like(img), np.zeros_like(img)

    from scipy.ndimage import convolve
    # based on https://dsp.stackexchange.com/questions/81517/how-to-find-the-kernel-of-the-convolution-for-linear-interpolation
    # adapted for irregular inputs
    from scipy.ndimage import distance_transform_edt
    unknown = np.isnan(img)
    distances = distance_transform_edt(unknown, return_distances=True)
    max_dist = int(np.ceil(distances.max()))  # this size is not adaptive and will cause over-smoothing in dense regions

    kernel_size_half = max_dist + 1
    hat = list(range(1, kernel_size_half + 1)) + list(range(kernel_size_half - 1, 0, -1))
    kernel_hat = np.array([hat])
    kernel = kernel_hat / kernel_hat.sum()  # separable kernel

    img[np.isnan(img)] = 0.0

    has_values = (img != 0).astype(np.float32)
    weights = convolve(has_values, kernel, mode='constant', cval=0.0)
    weights = convolve(weights, kernel.T, mode='constant', cval=0.0)

    values_sum = convolve(img, kernel, mode='constant', cval=0.0)
    values_sum = convolve(values_sum, kernel.T, mode='constant', cval=0.0)

    hm = values_sum / weights

    if np.isnan(hm).any():
        print('rasterized_to_linear produced NaNs')
        # No NaNs should be left here. If there are, it means that the kernel is not large enough.

    return hm, values_sum, weights


def gkern(l=5, sig=1.0):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    From: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l+1)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def rasterized_to_hqsplat(img: np.ndarray, kernel_size_half: int = 16):
    if np.all(np.isnan(img)):  # happens e.g. when no RGB is available
        return img, np.zeros_like(img), np.zeros_like(img)

    from scipy.ndimage import convolve
    # kernel = gkern(kernel_size_half, 1.5)  # for size_half = 10
    kernel = gkern(kernel_size_half, 2.25)  # for size_half = 16
    # kernel = gkern(kernel_size_half, 2.5)  # for size_half = 20

    img[np.isnan(img)] = 0.0

    has_values = (img != 0).astype(np.float32)
    weights = convolve(has_values, kernel, mode='constant', cval=0.0)
    values_sum = convolve(img, kernel, mode='constant', cval=0.0)
    hm = values_sum / weights
    hm[weights == 0.0] = 0.0  # fix NaNs

    return hm, values_sum, weights

