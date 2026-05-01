from ipes import cli_main
import argparse
import shlex
import sys

# run with:
# python ipes.py fit
# python ipes.py validate
# python ipes.py test
# python ipes.py predict
# configs as below

# profiling with tree visualization
# pip install snakeviz
# https://jiffyclub.github.io/snakeviz/
# python -m cProfile -o ipes.prof ipes.py
# snakeviz ipes.prof


print_cmd_only = False


COMMON_TEST_DATASETS = (
    'ca_13',
    'swisssurface3d',
    'ID15_Bunds',
    'Bund_BoraPk',
    'NZ23_Gisborne_subsets_BF44',
    'NZ23_Gisborne_subsets_BG41_0to23',
    'NZ23_Gisborne_subsets_BG41_24to50',
)


def make_configs(loss='rgb_mse', augmentation='enabled', use_valid_pixel_mask='enabled', gan=False):
    """Build config list for a run. Standard set: architecture, loss, training, dataset, augmentation, inputs."""
    configs = [
        'configs/architectures/cnn.yaml',
    ]
    if gan:
        configs.append('configs/architectures/gan.yaml')
    configs.extend([
        f'configs/losses/{loss}.yaml',
        f'configs/losses/use_valid_pixel_mask_{use_valid_pixel_mask}.yaml',
        'configs/training/default.yaml',
        'configs/datasets/train.yaml',
        f'configs/augmentation/{augmentation}.yaml',
        'configs/inputs/default.yaml',
        'configs/inputs/rgb_nearest_linear.yaml',
    ])
    return tuple(configs)


RUN_SPECS = (
    # {  # baseline
    #     'name': 'ipes_cnn_rgb',
    #     'configs': make_configs(),
    # },
    # {  # no significant difference
    #     'name': 'ipes_cnn_rgb_nomask',
    #     'configs': make_configs(use_valid_pixel_mask='disabled'),
    # },
    # {   # hm gradient is good
    #     'name': 'ipes_cnn_hm_mse_hm_gradient_rgb_mse_mask_noaug',
    #     'configs': make_configs(loss='hm_mse_hm_gradient_rgb_mse', augmentation='disabled') + ('configs/inputs/mask.yaml',),
    # },
    # {
    #     'name': 'ipes_cnn_hm_mse_rgb_mse_lpips_gradient_mask_noaug',
    #     'configs': make_configs(loss='hm_mse_rgb_mse_lpips_gradient', augmentation='disabled') + ('configs/inputs/mask.yaml',),
    # },
    # {   # one of the best
    #     'name': 'ipes_cnn_hm_mse_rgb_mse_lpips_ssim_gradient_learned_mask_noaug',
    #     'configs': make_configs(
    #         loss='hm_mse_rgb_mse_lpips_ssim_gradient_learned', augmentation='disabled') + ('configs/inputs/mask.yaml',),
    # },
    {
        'name': 'ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug',
        'configs': make_configs(
            loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', augmentation='disabled') + ('configs/inputs/mask.yaml',),
    },
    # {  # better without augmentation
    #     'name': 'ipes_cnn_rgb_noaug',
    #     'configs': make_configs(augmentation='disabled'),
    # },
    # {  # too slow, too bad
    #     'name': 'ipes_cnn_rgb_losses',
    #     'configs': make_configs(loss='rgb_advanced'),
    # },
    # {  # a bit better
    #     'name': 'ipes_cnn_rgb_mask',
    #     'configs': make_configs() + ('configs/inputs/mask.yaml',),
    # },
    # {   # bad results
    #     'name': 'ipes_cnn_rgb_fft',
    #     'configs': make_configs(loss='rgb_fft'),
    # },
    # {  # a bit worse, not matching metric
    #     'name': 'ipes_cnn_rgb_l1',
    #     'configs': make_configs(loss='rgb_l1'),
    # },
    # {
    #     'name': 'ipes_cnn_rgb_lpips',
    #     'configs': make_configs(loss='rgb_lpips'),
    # },
    # {
    #     'name': 'ipes_cnn_hm_mse_rgb_lpips',
    #     'configs': make_configs(loss='hm_mse_rgb_lpips'),
    # },
    # {
    #     'name': 'ipes_cnn_hm_mse_rgb_mse_lpips_learned',
    #     'configs': make_configs(loss='hm_mse_rgb_mse_lpips_learned'),
    # },
    # {   # surprisingly good
    #     'name': 'ipes_cnn_rgb_ssim',
    #     'configs': make_configs(loss='rgb_ssim'),
    # },
    # {   # takes forever, completely broken colors
    #     'name': 'ipes_cnn_rgb_flip',
    #     'configs': make_configs(loss='rgb_flip'),
    # },
    # {
    #     'name': 'ipes_gan_rgb',
    #     'configs': make_configs(gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_nomask',
    #     'configs': make_configs(use_valid_pixel_mask='disabled', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_noaug',
    #     'configs': make_configs(augmentation='disabled', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_losses',
    #     'configs': make_configs(loss='rgb_advanced', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_mask',
    #     'configs': make_configs(gan=True) + ('configs/inputs/mask.yaml',),
    # },
    # {   # bad results
    #     'name': 'ipes_gan_rgb_fft',
    #     'configs': make_configs(loss='rgb_fft', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_l1',
    #     'configs': make_configs(loss='rgb_l1', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_lpips',
    #     'configs': make_configs(loss='rgb_lpips', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_hm_mse_rgb_lpips',
    #     'configs': make_configs(loss='hm_mse_rgb_lpips', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_hm_mse_rgb_mse_lpips_learned',
    #     'configs': make_configs(loss='hm_mse_rgb_mse_lpips_learned', gan=True),
    # },
    # {
    #     'name': 'ipes_gan_rgb_ssim',
    #     'configs': make_configs(loss='rgb_ssim', gan=True),
    # },
    # {   # takes forever, not good
    #     'name': 'ipes_gan_rgb_flip',
    #     'configs': make_configs(loss='rgb_flip', gan=True),
    # },
    {
        'name': 'ipes_gan_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug',
        'configs': make_configs(
            loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', augmentation='disabled', gan=True) + ('configs/inputs/mask.yaml',),
    },
)


def build_config_args(config_paths):
    config_args = []
    for config_path in config_paths:
        config_args.extend(['-c', config_path])
    return config_args


def trainer_runtime_args():
    import torch

    # Keep script runnable on both GPU and CPU machines.
    return (
        ['--trainer.accelerator', 'gpu', '--trainer.devices', '-1']
        if torch.cuda.is_available()
        else ['--trainer.accelerator', 'cpu', '--trainer.devices', '1']
    )


def quote_cmd(argv):
    return ' '.join(shlex.quote(part) for part in argv)


def emit_cmd(argv, cmd=None):
    command = quote_cmd(argv)
    effective_print = print_cmd_only if cmd is None else bool(cmd)
    if effective_print:
        print(command)
        return

    sys.argv = argv
    cli_main()


def build_common_prefix(stage_name, name, config_paths):
    return [
        'ipes.py',
        stage_name,
        *build_config_args(config_paths),
        '--model.init_args.name', name,
        '--trainer.default_root_dir', f'models/{name}',
        *trainer_runtime_args(),
    ]


def build_fit_argv(spec):
    return [
        *build_common_prefix('fit', spec['name'], spec['configs']),
    ]


def build_stage_argv(stage_name, spec, dataset, input_template):
    overrides = (
        '--trainer.logger', 'False', '--data.init_args.workers', '0'
    ) if stage_name == 'predict' else ()
    
    return [
        *build_common_prefix(stage_name, spec['name'], spec['configs']),
        '--data.init_args.in_file', input_template.format(dataset=dataset),
        '--ckpt_path', f'models/{spec["name"]}/alpha/checkpoints/last.ckpt',
        *overrides,
    ]


def iter_run_argvs():
    for spec in RUN_SPECS:
        yield build_fit_argv(spec)

        for dataset in COMMON_TEST_DATASETS:
            yield build_stage_argv('test', spec, dataset, 'datasets/laz_minimal/test_{dataset}.txt')

        for dataset in COMMON_TEST_DATASETS:
            yield build_stage_argv('predict', spec, dataset, 'datasets/laz_minimal/bins/{dataset}/chunkPoints.csv')


def fixed_cmd(cmd=None):
    for argv in iter_run_argvs():
        emit_cmd(argv, cmd=cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_cmd_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    fixed_cmd(cmd=args.print_cmd_only)


if __name__ == '__main__':
    main()