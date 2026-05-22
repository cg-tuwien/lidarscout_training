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


def make_configs(
    loss='rgb_mse',
    augmentation='disabled',
    use_valid_pixel_mask='enabled',
    gan=False,
    dataset='train',
    include_rgb_input=True,
    include_mask=True,
):
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
        f'configs/datasets/{dataset}.yaml',
        f'configs/augmentation/{augmentation}.yaml',
        'configs/inputs/default.yaml',
    ])

    if include_rgb_input:
        configs.append('configs/inputs/rgb_nearest_linear.yaml')
    if include_mask:
        configs.append('configs/inputs/mask.yaml')
    
    return tuple(configs)


RUN_SPECS = (
    {   # Resume the best model for one more epoch to exercise last.pt export
        'name': 'ipes_cnn_v2_voidloss_arch2_resume_1epoch',
        'configs': make_configs(
            loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned'),
        'resume_ckpt_path': 'models/ipes_cnn_v2_voidloss_arch2/alpha/checkpoints/last.ckpt',
        'fit_overrides': ('--trainer.max_epochs', '76', '--data.init_args.workers', '4'),
    },
    # {   # Asymmetric Dual-Stream architecture, best so far
    #     'name': 'ipes_cnn_v2_voidloss', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned'),
    # },
    {   # Asymmetric Dual-Stream architecture, best so far
        'name': 'ipes_cnn_v2_voidloss_arch2', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
        'configs': make_configs(
            loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned'),
    },
    # {   # Asymmetric Dual-Stream architecture, slightly better PSNR but worse LPIPS
    #     'name': 'ipes_cnn_v2_norgbgrad', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
    #     'configs': make_configs(
    #         loss='hm_mse_radient_rgb_mse_lpips_learned'),
    # },
    # {   # Asymmetric Dual-Stream architecture, extra data ablation
    #     'name': 'ipes_cnn_v2_extra_data_voidloss', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', dataset='train_allstar'),
    # },
    # {   # Asymmetric Dual-Stream architecture, extra data ablation
    #     'name': 'ipes_cnn_v2_extra_data_voidloss', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', dataset='train_allstar'),
    # },
    {   # Asymmetric Dual-Stream architecture, extra data ablation
        'name': 'ipes_cnn_v2_extra_data_voidloss_arch2', # ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
        'configs': make_configs(
            loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', dataset='train_allstar'),
    },
    # {   # Asymmetric Dual-Stream architecture, colorization ablation
    #     'name': 'ipes_cnn_v2_colorizer',
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned',
    #         include_rgb_input=False),
    #     'overrides': ('--model.init_args.has_color_output', 'True'),
    # },
    # {   # Asymmetric Dual-Stream architecture, extra data ablation
    #     'name': 'ipes_cnn_v2_extra_data_nosundir',
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', dataset='train_allstar'),
    # },
    # {  # baseline
    #     'name': 'ipes_cnn_rgb',
    #     'configs': make_configs(),
    # },
    # {  # no significant difference
    #     'name': 'ipes_cnn_rgb_nomask',
    #     'configs': make_configs(use_valid_pixel_mask='disabled'),
    # },
    # {   # better without augmentation
    #     'name': 'ipes_cnn_rgb_noaug',
    #     'configs': make_configs(),
    # },
    # {   # hm gradient is good
    #     'name': 'ipes_cnn_hm_mse_hm_gradient_rgb_mse_mask_noaug',
    #     'configs': make_configs(loss='hm_mse_hm_gradient_rgb_mse'),
    # },
    # {
    #     'name': 'ipes_cnn_hm_mse_rgb_mse_lpips_gradient_mask_noaug',
    #     'configs': make_configs(loss='hm_mse_rgb_mse_lpips_gradient'),
    # },
    # {   # one of the best
    #     'name': 'ipes_cnn_hm_mse_rgb_mse_lpips_ssim_gradient_learned_mask_noaug',
    #     'configs': make_configs(
    #         loss='hm_mse_rgb_mse_lpips_ssim_gradient_learned'),
    # },
    # {
    #     'name': 'ipes_cnn_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug',
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned'),
    # },
    # {
    #     'name': 'ipes_gan_rgb',  # baseline + GAN
    #     'configs': make_configs(gan=True),
    # },
    # {
    #     'name': 'ipes_gan_v2',  # ipes_gan_hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned_mask_noaug with new CNN architecture
    #     'configs': make_configs(
    #         loss='hm_mse_hm_gradient_rgb_mse_lpips_gradient_learned', gan=True),
    # },
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


def build_common_prefix(stage_name, name, config_paths, overrides=()):
    return [
        'ipes.py',
        stage_name,
        *build_config_args(config_paths),
        '--model.init_args.name', name,
        *overrides,
        '--trainer.default_root_dir', f'models/{name}',
        *trainer_runtime_args(),
    ]


def build_fit_argv(spec):
    argv = [
        *build_common_prefix('fit', spec['name'], spec['configs'], spec.get('overrides', ())),
    ]

    resume_ckpt_path = spec.get('resume_ckpt_path')
    if resume_ckpt_path:
        argv.extend(['--ckpt_path', resume_ckpt_path])

    argv.extend(spec.get('fit_overrides', ()))
    return argv


def build_stage_argv(stage_name, spec, dataset, input_template):
    overrides = (
        '--trainer.logger', 'False', '--data.init_args.workers', '0'
    ) if stage_name == 'predict' else ()
    
    return [
        *build_common_prefix(stage_name, spec['name'], spec['configs'], spec.get('overrides', ())),
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