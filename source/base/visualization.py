import os
import typing

import numpy as np

from source.base import fs

if typing.TYPE_CHECKING:
    import torch


def images_to_figure(img_paths: typing.Sequence[str],
                     figure_path: str,
                     img_width: typing.Optional[int]=256,
                     img_height: typing.Optional[int]=256,
                     concat_dim_x=True):

    import PIL.Image

    # get img_width and img_height from first image
    if img_width is None or img_height is None:
        img = PIL.Image.open(img_paths[0])
        img_width, img_height = img.size

    # create figure
    width = img_width * len(img_paths) if concat_dim_x else img_width
    height = img_height if concat_dim_x else img_height * len(img_paths)
    figure = PIL.Image.new('RGB', (width, height), color=(255, 255, 255))

    # add images
    for i, img_path in enumerate(img_paths):
        img = PIL.Image.open(img_path)
        img = img.resize((img_width, img_height))
        paste_target = (i * img_width, 0) if concat_dim_x else (0, i * img_height)
        figure.paste(img, paste_target)

    # save figure
    fs.make_dir_for_file(figure_path)
    figure.save(figure_path)

    
def get_vis_file_aggregated(batch_data: dict, name, step, iteration, results_dir):
    base_name = os.path.join(results_dir, name, '00_all_{}'.format(name))

    if step in ['train', 'val']:
        arr_file = base_name + '_i{}'.format(iteration)
    elif step == 'test':
        batch_name: str = batch_data['pc_file_in'][0]
        batch_name = fs.sanitize_file_name(batch_name)
        arr_file = base_name + '_{}'.format(batch_name)
    elif step == 'predict':
        hm_id = batch_data['pts_query_ids_xy'][0]
        arr_file = base_name + '_id_{}_{}'.format(hm_id[0, 0].item(), hm_id[0, 1].item())
    else:
        raise NotImplementedError()
    return arr_file

    
def get_vis_file(batch_data: dict, name, step, iteration, results_dir, b):
    if step in ['train', 'val']:
        arr_file = os.path.join(results_dir, name, 'i{}_b{}_{}'.format(iteration, b, name))
    elif step == 'test':
        batch_name: str = batch_data['pc_file_in'][0]
        batch_name = fs.sanitize_file_name(batch_name)
        arr_file = os.path.join(results_dir, name, '{}_b{}_{}'.format(batch_name, b, name))
    elif step == 'predict':
        hm_id = batch_data['pts_query_ids_xy'][0]
        arr_file = os.path.join(results_dir, name, '_id{}_{}_b{}_{}'.format(
            hm_id[0, 0].item(), hm_id[0, 1].item(), b, name))
    else:
        raise NotImplementedError()
    return arr_file


def save_pts_local_ms(name, step, iteration,
                      batch_data, results_dir, vis_batches_range,
                      verbose=False, seed=42, color_factor=1.0):

    from source.base.normalization import patch_space_to_model_space_list
    from source.base.point_cloud import write_ply
    pts_local_ps = batch_data['pts_local_ps']
    pts_query_ms = batch_data['pts_query_ms'].detach().cpu().numpy()
    patch_radius = batch_data['patch_radius_interp_ms'].detach().cpu().numpy()
    # patch_radius_hm_ms = batch_data['patch_radius_hm_ms'].detach().cpu().numpy()
    numerical_stability_factor = batch_data['numerical_stability_factor'].detach().cpu().numpy()

    pts_ms_all = []
    colors_all = []
    rng = np.random.default_rng(seed)

    for b in vis_batches_range:
        pts_local_ps_batch = pts_local_ps[b].detach().cpu().numpy()
        pts_local_ms = patch_space_to_model_space_list(
            pts_to_convert_ps=pts_local_ps_batch, pts_patch_center_ms=pts_query_ms[b],
            patch_radius_ms=patch_radius[b], numerical_stability_z_factor=numerical_stability_factor[b])
        colors = np.broadcast_to(rng.random(3)[np.newaxis, :], pts_local_ms.shape) * color_factor
        pts_ms_all.append(pts_local_ms)
        colors_all.append(colors)

        if verbose:
            pts_file = os.path.join(results_dir, name, 'b{}_{}.ply'.format(b, name))
            fs.make_dir_for_file(pts_file)
            write_ply(file_path=pts_file, points=pts_local_ms, colors=colors[b])

    pts_ms_all = np.concatenate(pts_ms_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    pts_file_aggregated = get_vis_file_aggregated(batch_data, name, step, iteration, results_dir)
    fs.make_dir_for_file(pts_file_aggregated)
    write_ply(file_path=pts_file_aggregated, points=pts_ms_all, colors=colors_all)


def save_hm_as_pts(name, step, iteration,
                   batch_data, results_dir, vis_batches_range, hm_tensor_ps: 'torch.Tensor',
                   patch_radius: float, verbose=False, seed=42, color_factor=1.0, colors=None):
    from source.dataloaders.ipes_data_loader import hm_patch_space_to_model_space, hm_to_pts
    from source.base.point_cloud import write_ply
    hm = hm_tensor_ps.detach().cpu().numpy().astype(np.float32)
    pts_query_ms = batch_data['pts_query_ms'].detach().cpu().numpy()
    numerical_stability_factor = batch_data['numerical_stability_factor'].detach().cpu().numpy()

    pts_hm_ms_all = []
    pts_hm_norm_all = []
    colors_all = []
    rng = np.random.default_rng(seed)

    for b in vis_batches_range:
        hm_ms = hm_patch_space_to_model_space(
            hm_to_convert_ps=hm[b], pts_patch_center_ms=pts_query_ms[b],
            patch_radius_ms=patch_radius, numerical_stability_z_factor=numerical_stability_factor[0])
        pts_hm_ms, pts_hm_norm = hm_to_pts(hm_ms, pts_query_ms[b], pixel_size=10.0)
        if colors is not None:
            colors_patch = np.reshape(colors[b], (-1, 3))
            colors_patch[np.isnan(colors_patch)] = 0.0
        else:
            colors_patch = np.broadcast_to(rng.random(3)[np.newaxis], pts_hm_ms.shape) * color_factor
        pts_hm_ms_all.append(pts_hm_ms)
        pts_hm_norm_all.append(pts_hm_norm)
        colors_all.append(colors_patch)

        if verbose:
            pts_file = get_vis_file(batch_data, name, step, iteration, results_dir, b) + '.ply'
            fs.make_dir_for_file(pts_file)
            write_ply(file_path=pts_file, points=pts_hm_ms, colors=colors_patch)

    pts_hm_ms_all = np.concatenate(pts_hm_ms_all, axis=0)
    pts_hm_norm_all = np.concatenate(pts_hm_norm_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    pts_file = get_vis_file_aggregated(batch_data, name, step, iteration, results_dir) + '.ply'
    fs.make_dir_for_file(pts_file)
    write_ply(file_path=pts_file, points=pts_hm_ms_all, colors=colors_all, normals=pts_hm_norm_all)


def save_img_batch(batch_data: dict, arr, name,
             step, iteration, results_dir, vis_batches_range,
             norm_min=0.0, norm_max=1.0):
    from source.base.img import save_img

    arr_files = []
    for b in vis_batches_range:
        arr_file = get_vis_file(batch_data, name, step, iteration, results_dir, b) + '.png'
        save_img(arr=arr[b], file=arr_file, norm_min=norm_min, norm_max=norm_max)
        arr_files.append(arr_file)
    return arr_files
