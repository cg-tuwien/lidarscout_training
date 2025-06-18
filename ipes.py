import sys
import os
import typing

if typing.TYPE_CHECKING:
    from pytorch_lightning import cli

from source.cli import Cli

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


class IpesCli(Cli):

    def add_arguments_to_parser(self, parser: 'cli.LightningArgumentParser') -> None:
        super().add_arguments_to_parser(parser)

        parser.link_arguments('data.init_args.in_file', 'model.init_args.in_file')

        # this direction because logger is not available for test/predict
        parser.link_arguments('model.init_args.name', 'trainer.logger.init_args.name')

        # data inputs
        parser.link_arguments('data.init_args.hm_size', 'model.init_args.hm_size')
        parser.link_arguments('data.init_args.rgb_to_img_methods', 'model.init_args.rgb_to_img_methods')
        parser.link_arguments('data.init_args.pts_to_img_methods', 'model.init_args.pts_to_img_methods')

    def handle_rec_subcommand(self, args: typing.List[str]) -> typing.List[str]:
        """Replace 'rec' subcommand with predict and its default parameters.
        Download model if necessary.
        """

        # no rec -> nothing to do
        if len(args) <= 1 or args[1] != 'rec':
            return args

        # no rec -> nothing to do
        if len(args) <= 1 or args[1] != 'rec':
            return args

        # check syntax
        if len(args) < 4 or args[0] != os.path.basename(__file__):
            raise ValueError(
                'Invalid syntax for rec subcommand: {}\n'
                'Make sure that it matches this example: '
                'ipes.py rec in_file.ply out_file.ply --model.init_args.meters_per_pixel 10'.format(' '.join(sys.argv)))

        in_file = args[2]
        if not os.path.exists(in_file):
            raise ValueError('Input file does not exist: {}'.format(in_file))
        out_dir = args[3]
        os.makedirs(out_dir, exist_ok=True)
        extra_params = args[4:]
        model_path = os.path.join('models/ipes_cnn/version_0/checkpoints/last.ckpt')

        # assemble predict subcommand
        args_pred = args[:1]
        args_pred += [
            'predict',
            '-c', 'configs/ipes_cnn.yaml',
            '--ckpt_path', model_path,
            '--data.init_args.in_file', in_file,
            '--model.init_args.results_dir', out_dir,
            '--trainer.default_root_dir', 'models/ipes_cnn',
            '--trainer.logger', 'False',
            '--trainer.devices', '1'
            '--data.init_args.workers', '0',
            '--data.init_args.meters_per_pixel', '10',
        ]
        args_pred += extra_params
        print('Converted rec subcommand to predict subcommand: {}'.format(' '.join(args_pred)))

        # download model if necessary
        if not os.path.exists(model_path):
            print('Model checkpoint not found at {}. Downloading...'.format(model_path))
            os.system('python models/download_ppsurf_50nn.py')

        return args_pred


def cli_main():
    from source.base.nn import BaseModule
    from source.dataloaders.base_data_module import BaseDataModule
    IpesCli(model_class=BaseModule, subclass_mode_model=True,
            datamodule_class=BaseDataModule, subclass_mode_data=True)


def fixed_cmd():
    # name = 'ipes_cnn'
    name = 'ipes_cnn_rgb'
    # name = 'ipes_cnn_colorizer'
    # name = 'ipes_cnn_only_nn'
    # name = 'ipes_cnn_only_lin'
    # name = 'ipes_dctnet'
    # name = 'ipes_cnn_allstar'
    # name = 'ipes_unet'
    # name = 'ipes_rast'

    configs = [
        '-c', 'configs/ipes_cnn.yaml',
        '-c', 'configs/ipes_cnn_rgb.yaml',
        # '-c', 'configs/ipes_cnn_colorizer.yaml',
        # '-c', 'configs/ipes_cnn_only_nn.yaml',
        # '-c', 'configs/ipes_cnn_only_lin.yaml',
        # '-c', 'configs/ipes_dctnet.yaml',
        # '-c', 'configs/ipes_cnn_allstar.yaml',
        # '-c', 'configs/ipes_unet.yaml',
        # '-c', 'configs/ipes_rast.yaml',
        # '-c', 'configs/profiler.yaml'
    ]

    # train
    sys.argv = ['ipes.py',
                'fit',] + configs + [
                '--model.init_args.name', name,
                '--trainer.default_root_dir', 'models/{}'.format(name),
                # '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),  # to continue training
                # '--trainer.max_epochs', '151',
                # '--data.init_args.workers', '0',
                # '--trainer.max_epochs', '5',
                # '--debug', 'True',
                # '--print_config'
                ]
    print(' '.join(sys.argv))
    cli_main()

    # test
    test_sets_with_gt = [
        'ca_13',
        'swisssurface3d', 'ID15_Bunds',
        'Bund_BoraPk', 'NZ23_Gisborne_subsets_BF44',
        'NZ23_Gisborne_subsets_BG41_0to23', 'NZ23_Gisborne_subsets_BG41_24to50',
    ]
    for d in test_sets_with_gt:
        sys.argv = ['ipes.py',
                    'test',] + configs + [
                    '--data.init_args.in_file', 'datasets/laz_minimal/test_{}.txt'.format(d),
                    '--model.init_args.name', name,
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                    # '--print_config'
                    ]
        print(' '.join(sys.argv))
        cli_main()

    # predict
    test_sets_with_chunk_pts = test_sets_with_gt # + ['ahn5']
    for d in test_sets_with_chunk_pts:
        sys.argv = ['ipes.py',
                    'predict',] + configs + [
                    '--model.init_args.name', name,
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--data.init_args.in_file', 'datasets/laz_minimal/bins/{}/chunkPoints.csv'.format(d),
                    '--ckpt_path', 'models/{}/alpha/checkpoints/last.ckpt'.format(name),
                    '--trainer.logger', 'False',
                    '--data.init_args.workers', '0',
                    # '--data.init_args.meters_per_pixel', '1',
                    # '--data.init_args.meters_per_pixel', '2.5',
                    # '--data.init_args.meters_per_pixel', '5',
                    # '--data.init_args.meters_per_pixel', '10',  # default
                    # '--data.init_args.meters_per_pixel', '20',
                    # '--data.init_args.meters_per_pixel', '40',
                    # '--print_config'
                    ]
        print(' '.join(sys.argv))
        cli_main()


if __name__ == '__main__':
    fixed_cmd()
    # cli_main()
