import sys
import os
import typing

if typing.TYPE_CHECKING:
    from pytorch_lightning import cli

from source.cli import Cli


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
            '--trainer.devices', '1',
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


if __name__ == '__main__':
    cli_main()
