import os
import openpyxl as pyx
import numpy as np
import math
import typing
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

in_path = r'C:\repos\lidarscout_training\results'

datasets = [
    'ca_13', 
    # 'swisssurface3d', 'Bund_BoraPk', 'ID15_Bunds', 'NZ23_Gisborne_subsets_BF44',
    # 'NZ23_Gisborne_subsets_BG41_0to23', 'NZ23_Gisborne_subsets_BG41_24to50',
    # 'notoeast1', 'sitn',
            ]

runs = [
    'ipes_cnn_rgb',
    'ipes_gan',
    # 'ipes_cnn', 'ipes_cnn_colorizer', 'ipes_cnn_only_nn', 'ipes_cnn_only_lin', 'ipes_dctnet',
    # 'ipes_cnn_allstar', 'ipes_unet', 'ipes_rast',
    # 'ipes_interp_cubic', 'ipes_interp_linear', 'ipes_interp_rast_hqsplat_mean',
        ]

header = [
    'file',
    'hm_rmse_ms',
    'rgb_psnr',
    'rgb_gradient_rmse',
]

join_str = '\t '
value_format_str = '{:.4f}'

# header
header_joined = join_str.join(header)
print(header_joined)

# image stitching config
image_rel_dir = os.path.join('laz_minimal', 'test', '00_hm_rgb_fig_input')
image_file_names = [
    'ca_13_0_10_b0_00_hm_rgb_fig_input.png',
]


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    raise ValueError(f'Unexpected image shape: {img.shape}')


def _crop_patch_figure(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    target_w = min(w, h)
    target_h = max(1, h // 2)

    x0 = max(0, w - target_w)
    y0 = max(0, h - target_h)

    cropped = img[y0:y0 + target_h, x0:x0 + target_w]
    return cropped


def _add_label(img: np.ndarray, lines: typing.List[str], pad: int = 6) -> np.ndarray:
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    line_heights = []
    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        max_line_w = max(max_line_w, line_w)
        line_heights.append(line_h)

    text_h = int(sum(line_heights) + max(0, len(lines) - 1) * 2)
    header_h = text_h + 2 * pad
    out_w = max(pil_img.width, max_line_w + 2 * pad)

    out = Image.new('RGB', (out_w, pil_img.height + header_h), color=(255, 255, 255))
    out.paste(pil_img, (0, header_h))

    draw_out = ImageDraw.Draw(out)
    y = pad
    for line, line_h in zip(lines, line_heights):
        draw_out.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h + 2

    return np.asarray(out)


def _metric_bg_color(value: typing.Optional[float], vmin: float, vmax: float, higher_is_better: bool) -> typing.Tuple[int, int, int]:
    if value is None:
        return 255, 255, 255

    if vmax <= vmin:
        score = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))
        score = norm if higher_is_better else 1.0 - norm

    red = int(round(255.0 * (1.0 - score)))
    green = int(round(255.0 * score))
    blue = 140
    return red, green, blue


def _load_monospace_font(size: int = 14) -> ImageFont.ImageFont:
    font_candidates = [
        r'C:\Windows\Fonts\consola.ttf',
        r'C:\Windows\Fonts\consolab.ttf',
        'DejaVuSansMono.ttf',
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fmt_fixed(value: typing.Optional[float], width: int = 8, precision: int = 4) -> str:
    if value is None:
        return 'None'.rjust(width)
    return f'{value:>{width}.{precision}f}'


def _ellipsize_left(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ''
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[-max_len:]
    return '...' + text[-(max_len - 3):]


def _add_label_colored_metrics(
        img: np.ndarray,
        run_name: str,
        means: typing.Dict[str, typing.Optional[float]],
        metric_ranges: typing.Dict[str, typing.Tuple[float, float]],
        run_name_width: int,
        value_fmt: str,
    pad: int = 2) -> np.ndarray:
    pil_img = Image.fromarray(img)

    hm_val = means.get('hm_rmse_ms_mean')
    psnr_val = means.get('rgb_psnr_mean')
    grad_val = means.get('rgb_gradient_rmse_mean')

    # Known ranges keep columns compact while preserving fixed-width alignment.
    hm_width = max(len('9.9999'), len('None'))
    psnr_width = max(len('40.0000'), len('None'))
    grad_width = max(len('0.0000'), len('None'))
    hm_text = _fmt_fixed(hm_val, width=hm_width, precision=4)
    psnr_text = _fmt_fixed(psnr_val, width=psnr_width, precision=4)
    grad_text = _fmt_fixed(grad_val, width=grad_width, precision=4)

    selected_font = None
    selected_run_text = None
    selected_metric_segments = None
    selected_run_line_w = None
    selected_metric_seg_sizes = None
    selected_metric_line_w = None
    selected_line_h = None

    # Fit both lines into image width: first shrink run name, then font size if needed.
    for font_size in [14, 13, 12, 11, 10, 9, 8]:
        font = _load_monospace_font(size=font_size)
        draw_tmp = ImageDraw.Draw(Image.new('RGB', (1, 1), color=(255, 255, 255)))

        metrics_left = 'hm_rmse_ms='
        metrics_mid = '|rgb_psnr='
        metrics_right = '|rgb_gradient_rmse='
        run_len = run_name_width
        while run_len >= 1:
            run_text = _ellipsize_left(run_name, run_len).rjust(run_len)
            run_bbox = draw_tmp.textbbox((0, 0), run_text, font=font)
            run_line_w = int(run_bbox[2] - run_bbox[0])

            metric_segments: typing.List[typing.Tuple[str, typing.Optional[typing.Tuple[int, int, int]]]] = [
                (metrics_left, None),
                (hm_text, _metric_bg_color(hm_val, *metric_ranges['hm_rmse_ms_mean'], higher_is_better=False)),
                (metrics_mid, None),
                (psnr_text, _metric_bg_color(psnr_val, *metric_ranges['rgb_psnr_mean'], higher_is_better=True)),
                (metrics_right, None),
                (grad_text, _metric_bg_color(grad_val, *metric_ranges['rgb_gradient_rmse_mean'], higher_is_better=False)),
            ]

            metric_seg_sizes = []
            for text, _ in metric_segments:
                bbox = draw_tmp.textbbox((0, 0), text, font=font)
                metric_seg_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

            metric_line_w = int(sum(w for w, _ in metric_seg_sizes))
            line_h = int(max([run_bbox[3] - run_bbox[1]] + [h for _, h in metric_seg_sizes]))
            if max(run_line_w, metric_line_w) <= (pil_img.width - 2 * pad):
                selected_font = font
                selected_run_text = run_text
                selected_metric_segments = metric_segments
                selected_run_line_w = run_line_w
                selected_metric_seg_sizes = metric_seg_sizes
                selected_metric_line_w = metric_line_w
                selected_line_h = line_h
                break

            run_len -= 1

        if selected_font is not None:
            break

    if selected_font is None:
        selected_font = _load_monospace_font(size=8)
        selected_run_text = _ellipsize_left(run_name, max(1, run_name_width // 2))
        selected_metric_segments = [
            ('h=', None),
            (hm_text, _metric_bg_color(hm_val, *metric_ranges['hm_rmse_ms_mean'], higher_is_better=False)),
            ('|p=', None),
            (psnr_text, _metric_bg_color(psnr_val, *metric_ranges['rgb_psnr_mean'], higher_is_better=True)),
            ('|g=', None),
            (grad_text, _metric_bg_color(grad_val, *metric_ranges['rgb_gradient_rmse_mean'], higher_is_better=False)),
        ]
        draw_tmp = ImageDraw.Draw(Image.new('RGB', (1, 1), color=(255, 255, 255)))
        run_bbox = draw_tmp.textbbox((0, 0), selected_run_text, font=selected_font)
        selected_run_line_w = int(run_bbox[2] - run_bbox[0])
        selected_metric_seg_sizes = []
        for text, _ in selected_metric_segments:
            bbox = draw_tmp.textbbox((0, 0), text, font=selected_font)
            selected_metric_seg_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        selected_metric_line_w = int(sum(w for w, _ in selected_metric_seg_sizes))
        selected_line_h = int(max([run_bbox[3] - run_bbox[1]] + [h for _, h in selected_metric_seg_sizes]))

    line_h = selected_line_h
    header_h = 2 * line_h + 3 * pad
    out_w = pil_img.width

    out = Image.new('RGB', (out_w, pil_img.height + header_h), color=(255, 255, 255))
    img_x = out_w - pil_img.width
    out.paste(pil_img, (img_x, header_h))

    draw_out = ImageDraw.Draw(out)
    run_x = out_w - pad - selected_run_line_w
    run_y = pad
    draw_out.text((run_x, run_y), selected_run_text, fill=(0, 0, 0), font=selected_font)

    x = out_w - pad - selected_metric_line_w
    y = pad * 2 + line_h
    for (text, bg_color), (seg_w, seg_h) in zip(selected_metric_segments, selected_metric_seg_sizes):
        if bg_color is not None:
            draw_out.rectangle([x, y, x + seg_w, y + line_h], fill=bg_color)
        draw_out.text((x, y), text, fill=(0, 0, 0), font=selected_font)
        x += seg_w

    return np.asarray(out)


def _make_grid(images: typing.List[np.ndarray], pad: int = 10, bg_value: int = 255) -> np.ndarray:
    if len(images) == 0:
        raise ValueError('No images to stitch')

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    grid_h = rows * max_h + (rows + 1) * pad
    grid_w = cols * max_w + (cols + 1) * pad
    grid = np.full((grid_h, grid_w, 3), bg_value, dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = pad + row * (max_h + pad)
        x0 = pad + col * (max_w + pad)

        y_off = (max_h - img.shape[0]) // 2
        x_off = 0

        ys = y0 + y_off
        xs = x0 + x_off
        grid[ys:ys + img.shape[0], xs:xs + img.shape[1], :] = img

    return grid


def _get_metric_means(excel_file: str) -> typing.Dict[str, typing.Optional[float]]:
    wb = pyx.load_workbook(excel_file)
    ws = wb.active

    has_color = ws['F1'].value is not None
    if has_color:
        ranges_to_get = ['E2:E31', 'G2:G31', 'H2:H31']
        metric_names = ['hm_rmse_ms_mean', 'rgb_psnr_mean', 'rgb_gradient_rmse_mean']
    else:
        ranges_to_get = ['D2:D31']
        metric_names = ['hm_rmse_ms_mean']

    means: typing.Dict[str, typing.Optional[float]] = {}
    for i, range_to_get in enumerate(ranges_to_get):
        cells = [c for c in ws[range_to_get]]
        cell_vals = [c[0].value for c in cells]

        # fix broken PSNR
        if i == 1:
            cell_vals = [v - 48.131 if v is not None and v > 30 else v for v in cell_vals]

        if all(v is None for v in cell_vals):
            means[metric_names[i]] = None
            continue

        cell_vals_np = np.asarray(cell_vals, dtype=np.float64)
        means[metric_names[i]] = float(np.mean(cell_vals_np))

    wb.close()
    return means


metrics_by_run: typing.Dict[str, typing.Dict[str, typing.Optional[float]]] = {}
metrics_by_run_dataset: typing.Dict[typing.Tuple[str, str], typing.Dict[str, typing.Optional[float]]] = {}

for dataset in datasets:
    # get all excel files recursively
    excel_files = []
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file.endswith('.xlsx') and file.find(dataset) != -1:
                contains_run = any([file.find(run) != -1 for run in runs])
                if contains_run:
                    excel_files.append(os.path.join(root, file))

    # read all excel files
    for excel_file in excel_files:
        try:
            means = _get_metric_means(excel_file)
        except Exception:
            print(f'Error loading {excel_file}')
            continue

        vals_to_print = []
        vals_to_print.append(value_format_str.format(means['hm_rmse_ms_mean']) if means['hm_rmse_ms_mean'] is not None else 'None')
        vals_to_print.append(value_format_str.format(means['rgb_psnr_mean']) if 'rgb_psnr_mean' in means and means['rgb_psnr_mean'] is not None else 'None')
        vals_to_print.append(value_format_str.format(means['rgb_gradient_rmse_mean']) if 'rgb_gradient_rmse_mean' in means and means['rgb_gradient_rmse_mean'] is not None else 'None')

        run_name = os.path.basename(excel_file).replace('metrics_', '').split('_test_')[0]
        metrics_by_run[run_name] = means
        metrics_by_run_dataset[(run_name, dataset)] = means

        vals_joined = join_str.join(vals_to_print)
        print(f'{os.path.basename(excel_file)[13:]} {join_str} {vals_joined}', flush=True)

# build labeled patch comparison images per dataset
run_dirs = sorted([d for d in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, d))])
for dataset in datasets:
    dataset_image_files = [f for f in image_file_names if f.startswith(dataset + '_')]
    if len(dataset_image_files) == 0:
        print(f'No image file pattern configured for dataset {dataset}')
        continue

    for image_file_name in dataset_image_files:
        run_entries = []
        for run_dir in run_dirs:
            if not any(run in run_dir for run in runs):
                continue

            img_file = os.path.join(in_path, run_dir, image_rel_dir, image_file_name)
            if not os.path.isfile(img_file):
                continue

            try:
                img = iio.imread(img_file)
                img = _to_rgb(np.asarray(img))
                img = _crop_patch_figure(img)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                means = metrics_by_run_dataset.get((run_dir, dataset), metrics_by_run.get(run_dir, {}))
                if means is None:
                    means = {}
                run_entries.append((run_dir, img, means))
            except Exception as exc:
                print(f'Error loading image {img_file}: {exc}')

        images_labeled = []
        if len(run_entries) > 0:
            metric_ranges = {}
            run_name_width = max(len(entry[0]) for entry in run_entries)
            for metric_name in ['hm_rmse_ms_mean', 'rgb_psnr_mean', 'rgb_gradient_rmse_mean']:
                vals = [entry[2].get(metric_name) for entry in run_entries]
                vals = [v for v in vals if v is not None]
                if len(vals) == 0:
                    metric_ranges[metric_name] = (0.0, 1.0)
                else:
                    metric_ranges[metric_name] = (float(min(vals)), float(max(vals)))

            for run_dir, img, means in run_entries:
                labeled = _add_label_colored_metrics(
                    img=img,
                    run_name=run_dir,
                    means=means,
                    metric_ranges=metric_ranges,
                    run_name_width=run_name_width,
                    value_fmt=value_format_str,
                )
                images_labeled.append(labeled)

        if len(images_labeled) > 0:
            stitched = _make_grid(images_labeled)
            image_file_stem, _ = os.path.splitext(image_file_name)
            image_out_file = os.path.join(in_path, f'patch_comp_{image_file_stem}.png')
            iio.imwrite(image_out_file, stitched)
            print(f'Wrote {image_out_file} ({len(images_labeled)} images)')
        else:
            print(f'No patch images found for dataset={dataset}, file={image_file_name}.')