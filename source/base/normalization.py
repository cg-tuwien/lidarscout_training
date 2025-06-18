import numbers
import typing
if typing.TYPE_CHECKING:
    import torch

import numpy as np


def model_space_to_patch_space_nmp(
        pts_to_convert_ms: np.array,                       # (q, k, 3)
        pts_patch_center_ms: np.array,                     # (q, 3)
        patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
        numerical_stability_z_factor: float = 10.0):

    patch_radius_ms_repeated = np.broadcast_to(patch_radius_ms, pts_to_convert_ms.shape)
    pts_patch_center_ms_repeated = np.broadcast_to(pts_patch_center_ms[:, np.newaxis], pts_to_convert_ms.shape)

    pts_patch_space = pts_to_convert_ms - pts_patch_center_ms_repeated
    pts_patch_space = pts_patch_space / patch_radius_ms_repeated
    pts_patch_space[..., 2] *= numerical_stability_z_factor
    return pts_patch_space

def patch_space_to_model_space_nmp(
        pts_to_convert_ps: np.array,                       # (q, k, 3)
        pts_patch_center_ms: np.array,                     # (q, 3)
        patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
        numerical_stability_z_factor: float = 10.0):

    pts_patch_center_ms_repeated = np.broadcast_to(pts_patch_center_ms, pts_to_convert_ps.shape)
    patch_radius_ms_repeated = np.broadcast_to(patch_radius_ms, pts_to_convert_ps.shape)

    pts_to_convert_ps[..., 2] /= numerical_stability_z_factor
    pts_model_space = pts_to_convert_ps * patch_radius_ms_repeated
    pts_model_space = pts_model_space + pts_patch_center_ms_repeated
    return pts_model_space


def hm_model_space_to_patch_space(hm_to_convert_ms: np.array,                        # (q, s, s)
                                  pts_patch_center_ms: np.array,                     # (q, 3)
                                  patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
                                  numerical_stability_z_factor: float = 10.0):

    patch_mean_z_ms = pts_patch_center_ms[..., 2]  # (q,)
    patch_mean_z_ms_repeated = np.broadcast_to(patch_mean_z_ms[:, None, None], hm_to_convert_ms.shape)
    patch_radius_ms_repeated = np.broadcast_to(patch_radius_ms, hm_to_convert_ms.shape)

    hm_ms_patch_space = hm_to_convert_ms - patch_mean_z_ms_repeated
    hm_ms_patch_space = hm_ms_patch_space / patch_radius_ms_repeated

    hm_ms_patch_space *= numerical_stability_z_factor
    return hm_ms_patch_space


def hm_patch_space_to_model_space(hm_to_convert_ps: np.array,                        # (q, s, s)
                                  pts_patch_center_ms: np.array,                     # (q, 3)
                                  patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
                                  numerical_stability_z_factor: float = 10.0):

    patch_mean_z_ms = pts_patch_center_ms[..., 2]  # (q,)
    patch_mean_z_ms_repeated = np.broadcast_to(patch_mean_z_ms, hm_to_convert_ps.shape)
    patch_radius_ms_repeated = np.broadcast_to(patch_radius_ms, hm_to_convert_ps.shape)

    hm_to_convert_ps /= numerical_stability_z_factor

    hm_model_space = hm_to_convert_ps * patch_radius_ms_repeated
    hm_model_space = hm_model_space + patch_mean_z_ms_repeated
    return hm_model_space


def hm_patch_space_to_model_space_tensor(hm_to_convert_ps: 'torch.Tensor',                        # (q, s, s)
                                         pts_patch_center_ms: 'torch.Tensor',                     # (q, 3)
                                         patch_radius_ms: typing.Union['torch.Tensor', float],    # (q)
                                         numerical_stability_z_factor: float = 10.0):

    import torch

    if isinstance(patch_radius_ms, numbers.Number):
        patch_radius_ms = torch.as_tensor(patch_radius_ms, device=hm_to_convert_ps.device)

    patch_mean_z_ms = pts_patch_center_ms[..., 2]  # (q,)
    patch_mean_z_ms_repeated = patch_mean_z_ms[:, None, None].expand_as(hm_to_convert_ps)
    patch_radius_ms_repeated = patch_radius_ms.expand_as(hm_to_convert_ps)

    hm_to_convert_ps = hm_to_convert_ps / numerical_stability_z_factor

    hm_model_space = hm_to_convert_ps * patch_radius_ms_repeated
    hm_model_space = hm_model_space + patch_mean_z_ms_repeated
    return hm_model_space


def model_space_to_patch_space_list(pts_to_convert_ms: typing.List[np.ndarray],        # (q)(?, 3)
                                    pts_patch_center_ms: np.ndarray,                   # (q, 3)
                                    patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
                                    numerical_stability_z_factor: float = 10.0):

    def _convert_single_patch(pts_patch_ms, pts_patch_center_ms, patch_radius_ms):
        pts_patch_ps = model_space_to_patch_space_nmp(
            pts_to_convert_ms=pts_patch_ms[np.newaxis],
            pts_patch_center_ms=pts_patch_center_ms[np.newaxis],
            patch_radius_ms=np.asarray(patch_radius_ms)[np.newaxis],
            numerical_stability_z_factor=numerical_stability_z_factor)[0]
        return pts_patch_ps

    pts_patch_space = [_convert_single_patch(pts, pts_patch_center_ms[i], patch_radius_ms)
                       for i, pts in enumerate(pts_to_convert_ms)]
    return pts_patch_space


def patch_space_to_model_space_list(pts_to_convert_ps: typing.List[np.ndarray],        # (q)(?, 3)
                                    pts_patch_center_ms: np.ndarray,                   # (q, 3)
                                    patch_radius_ms: typing.Union[np.ndarray, float],  # (q)
                                    numerical_stability_z_factor: float = 10.0):

    def _convert_single_patch(pts_patch_ps, pts_patch_center_ms, patch_radius_ms):
        pts_patch_ms = patch_space_to_model_space_nmp(
            pts_to_convert_ps=pts_patch_ps[np.newaxis],
            pts_patch_center_ms=pts_patch_center_ms[np.newaxis],
            patch_radius_ms=patch_radius_ms[np.newaxis],
            numerical_stability_z_factor=numerical_stability_z_factor)[0]
        return pts_patch_ms

    pts_model_space = [_convert_single_patch(pts, pts_patch_center_ms[i], patch_radius_ms)
                       for i, pts in enumerate(pts_to_convert_ps)]
    return pts_model_space


def _test_normalize():
    ms = 0.75
    vs = 1.0 / 32
    # padding_factor = 0.0
    padding_factor = 0.05
    pts_ms = np.array([[-ms, -ms], [-ms, +ms], [+ms, -ms], [+ms, +ms], [0.0, 0.0],
                       [vs*0.3, -vs*0.3], [vs*0.5, -vs*0.5], [vs*0.6, -vs*0.6]])
    pts_ms *= 76.0
    pts_ms += 123.0

    # vertices = np.random.random(size=(25, 2)) * 2.0 - 1.0
    vertices = pts_ms

    bb_center, scale = get_points_normalization_info(pts=pts_ms, padding_factor=padding_factor)
    vertices_norm = normalize_points_with_info(pts=vertices, bb_center=bb_center, scale=scale)
    vertices_denorm = denormalize_points_with_info(pts=vertices_norm, bb_center=bb_center, scale=scale)

    if not np.allclose(vertices_denorm, vertices):
        raise ValueError()

    if vertices_norm.max() > 0.5 or vertices_norm.min() < -0.5:
        raise ValueError()

    return 0


def _unit_test_ms_ps():
    patch_radius_ms = np.full(10, 0.1)
    numerical_stability_factor = 10.0

    np.random.seed(42)
    pts_ms = np.random.rand(10, 50, 3)
    pts_query_ms = np.random.rand(10, 3)

    pts_ps = model_space_to_patch_space_nmp(
        pts_to_convert_ms=pts_ms, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_ms,
        numerical_stability_z_factor=numerical_stability_factor)
    pts_ms_rec = patch_space_to_model_space_nmp(
        pts_to_convert_ps=pts_ps, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_ms,
        numerical_stability_z_factor=numerical_stability_factor)

    if not np.allclose(pts_ms, pts_ms_rec):
        raise ValueError()


def _unit_test_hm_ms_ps():
    patch_radius_ms = np.full(10, 0.1)
    numerical_stability_factor = 10.0

    np.random.seed(42)
    hm_ms = np.random.rand(10, 64, 64)
    pts_query_ms = np.random.rand(10, 3)

    hm_ps = hm_model_space_to_patch_space(
        hm_to_convert_ms=hm_ms, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_ms,
        numerical_stability_z_factor=numerical_stability_factor)
    hm_ms_rec = hm_patch_space_to_model_space(
        hm_to_convert_ps=hm_ps, pts_patch_center_ms=pts_query_ms, patch_radius_ms=patch_radius_ms,
        numerical_stability_z_factor=numerical_stability_factor)

    if not np.allclose(hm_ms, hm_ms_rec):
        raise ValueError()


if __name__ == '__main__':
    _test_normalize()
    _unit_test_ms_ps()
    _unit_test_hm_ms_ps()
