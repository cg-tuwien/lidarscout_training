import sys

# run with:
# python ipes_interp.py test
# python ipes_interp.py predict
# configs as below

# profiling with tree visualization
# pip install snakeviz
# https://jiffyclub.github.io/snakeviz/
# python -m cProfile -o ipes_interp.prof ipes_interp.py
# snakeviz ipes_interp.prof

from ipes import IpesCli


class IpesInterpCli(IpesCli):
    """
    disable optimizer so we can skip any NN and training
    """

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        # self._add_configure_optimizers_method_to_model(self.subcommand)  # no optimizer for simple interpolation
        self.trainer = self.instantiate_trainer()


def cli_main():
    from source.base.nn import BaseModule
    from source.dataloaders.base_data_module import BaseDataModule
    IpesInterpCli(model_class=BaseModule, subclass_mode_model=True,
                  datamodule_class=BaseDataModule, subclass_mode_data=True)


def fixed_cmd():
    methods = ['rast_hqsplat_mean', 'linear', 'cubic']
    # methods = ['nearest']
    for method in methods:
        name = 'ipes_interp_{}'.format(method)

        # no train
        pass

        # test
        test_sets_with_gt = [
            'ca_13', 'swisssurface3d',
            'ID15_Bunds', 'Bund_BoraPk', 'NZ23_Gisborne_subsets_BF44',
            'NZ23_Gisborne_subsets_BG41_0to23', 'NZ23_Gisborne_subsets_BG41_24to50'
        ]
        for d in test_sets_with_gt:
            sys.argv = ['ipes_interp.py',
                        'test',
                        '-c', 'configs/ipes_interp.yaml',
                        '--trainer.default_root_dir', 'models/{}'.format(name),
                        '--model.init_args.name', name,
                        '--data.init_args.in_file', 'datasets/laz_minimal/test_{}.txt'.format(d),
                        '--data.init_args.pts_to_img_methods', '[{}]'.format(method),
                        '--data.init_args.rgb_to_img_methods', '[{}]'.format(method),
                        # '--print_config'
                        ]
            print(' '.join(sys.argv))
            cli_main()

        # predict
        test_sets_with_chunk_pts = test_sets_with_gt # + ['ahn5']
        for d in test_sets_with_chunk_pts:
            sys.argv = ['ipes_interp.py',
                        'predict',
                        '-c', 'configs/ipes_interp.yaml',
                        '--trainer.default_root_dir', 'models/{}'.format(name),
                        '--trainer.logger', 'False',
                        '--data.init_args.pts_to_img_methods', '[{}]'.format(method),
                        '--data.init_args.rgb_to_img_methods', '[{}]'.format(method),
                        '--model.init_args.name', name,
                        '--data.init_args.in_file', 'datasets/laz_minimal/bins/{}/chunkPoints.csv'.format(d),
                        # '--data.init_args.workers', '0',
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
    # fixed_cmd()
    cli_main()
