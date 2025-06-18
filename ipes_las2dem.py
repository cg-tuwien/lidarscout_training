import sys

# run with:
# python ipes_las2dem.py test
# python ipes_las2dem.py predict
# configs as below

# profiling with tree visualization
# pip install snakeviz
# https://jiffyclub.github.io/snakeviz/
# python -m cProfile -o ipes_las2dem.prof ipes_las2dem.py
# snakeviz ipes_las2dem.prof

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
    name = 'ipes_las2dem'

    # no train
    pass

    # test
    test_sets_with_gt = [
        'ca_13', 'swisssurface3d', 'ID15_Bunds',
        'Bund_BoraPk', 'NZ23_Gisborne_subsets_BF44',
        'NZ23_Gisborne_subsets_BG41_0to23', 'NZ23_Gisborne_subsets_BG41_24to50']
    for d in test_sets_with_gt:
        sys.argv = ['ipes_las2dem.py',
                    'test',
                    '-c', 'configs/ipes_las2dem.yaml',
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--model.init_args.name', name,
                    '--data.init_args.in_file', 'datasets/laz_minimal/test_set_{}.txt'.format(d),
                    # '--print_config'
                    ]
        cli_main()

    # predict
    test_sets_with_chunk_pts = test_sets_with_gt + ['ahn5']
    for d in test_sets_with_chunk_pts:
        sys.argv = ['ipes_las2dem.py',
                    'predict',
                    '-c', 'configs/ipes_las2dem.yaml',
                    '--trainer.default_root_dir', 'models/{}'.format(name),
                    '--trainer.logger', 'False',
                    '--model.init_args.name', name,
                    '--data.init_args.in_file', 'datasets/laz_minimal/bins/{}/chunkPoints.csv'.format(d),
                    # '--print_config'
                    ]
        cli_main()


if __name__ == '__main__':
    fixed_cmd()
    # cli_main()
