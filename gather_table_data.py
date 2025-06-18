import os
import openpyxl as pyx
import numpy as np

in_path = r'E:\repos_ssd\ipes\results'

datasets = [
    'ca_13', 'swisssurface3d', 'Bund_BoraPk', 'ID15_Bunds', 'NZ23_Gisborne_subsets_BF44',
    'NZ23_Gisborne_subsets_BG41_0to23', 'NZ23_Gisborne_subsets_BG41_24to50',
    # 'notoeast1', 'sitn',
            ]

runs = [
    'ipes_cnn_rgb',
    'ipes_cnn', 'ipes_cnn_colorizer', 'ipes_cnn_only_nn', 'ipes_cnn_only_lin', 'ipes_dctnet',
    'ipes_cnn_allstar', 'ipes_unet', 'ipes_rast',
    'ipes_interp_cubic', 'ipes_interp_linear', 'ipes_interp_rast_hqsplat_mean',
        ]

header = [
    'file',
    'abs_dist_rmse_ms_mean', 'abs_dist_rmse_ms_std',
    'rgb_psnr_mean', 'rgb_psnr_std',
    'rgb_lpips_mean', 'rgb_lpips_std'
]
header_joined = '\t '.join(header)
print(header_joined)

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
            wb = pyx.load_workbook(excel_file)
        except:
            print(f'Error loading {excel_file}')
            continue
        ws = wb.active

        vals_to_print = []

        has_color = ws['F1'].value is not None
        if has_color:
            ranges_to_get = ['E2:E31', 'H2:H31', 'I2:I31']
        else:
            ranges_to_get = ['D2:D31']

        # get range values
        for range_to_get in ranges_to_get:
            cells = [c for c in ws[range_to_get]]

            cell_vals = [c[0].value for c in cells]
            if all([v is None for v in cell_vals]):
                vals_to_print.append('None')
                vals_to_print.append('None')
                continue

            cell_vals_np = np.array(cell_vals)
            cells_mean = np.mean(cell_vals_np)
            cells_std = np.std(cell_vals_np)
            vals_to_print.append(str(cells_mean))
            vals_to_print.append(str(cells_std))

        vals_joined = '\t '.join(vals_to_print)
        print(f'{os.path.basename(excel_file)[13:]}\t {vals_joined}', flush=True)
        wb.close()