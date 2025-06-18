import os
import sys
import typing
import abc

from tqdm import tqdm

from pytorch_lightning.cli import LightningCLI, Namespace, LightningArgumentParser
from pytorch_lightning.callbacks import TQDMProgressBar

# for derived ModelCheckpoint class
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.types import _PATH
from typing import Optional, Literal, Dict
from datetime import timedelta
from torch import Tensor

from source.base.profiling import get_now_str


class PPSProgressBar(TQDMProgressBar):  # disable validation prog bar
    def init_validation_tqdm(self):
        bar_disabled = tqdm(disable=True)
        return bar_disabled


class TorchScriptModelCheckpoint(ModelCheckpoint):

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Literal[True, False, "link"]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        from source.base.fs import make_dir_for_file

        # return  # disable overwriting for debugging
        super()._save_last_checkpoint(trainer, monitor_candidates)

        if trainer.current_epoch != trainer.max_epochs - 1:
            return  # only save TorchScript model at the end of training

        network = trainer.lightning_module

        # does not work with LightningModule forward() having *args and **kwargs
        # net_ts = network.to_torchscript(model_script_path, method='script')

        # save with tracing, need a training sample that will be traced through the forward pass
        def make_example_data():
            import torch
            res = network.hm_interp_size
            example_inputs = dict()

            # not required
            # example_inputs['pc_file_in'] = ['hms.bin,3900,4000']  # no list for tracing
            # example_inputs['hm_gt_ps'] = torch.rand(2, 64, 64)
            # example_inputs['pts_local_ms'] = torch.rand(2, 500, 3)
            # example_inputs['pts_local_ps'] = torch.rand(2, 500, 3)
            # example_inputs['pts_local_ms_z_mean'] = torch.rand(2)
            # example_inputs['hm_gt_ps'] = torch.rand(2, 64, 64)
            # example_inputs['patch_radius_interp_ms'] = torch.tensor(data=[678.8225], dtype=torch.float32)

            # required for model-space output
            # example_inputs['pts_query_ms'] = torch.rand(2, 3)
            # example_inputs['patch_radius_hm_ms'] = torch.tensor(data=[678.8225], dtype=torch.float32)
            # example_inputs['numerical_stability_factor'] = torch.rand(1, )

            # required for DCTNet  # TODO: DCTNet has no network.input_methods
            for m in network.input_methods:
                example_inputs[f'patch_hm_{m}'] = torch.rand(2, 1, res, res)
                example_inputs[f'patch_rgb_{m}'] = torch.rand(2, 3, res, res)
            return example_inputs

        # model_script_path_onnx = os.path.join(self.dirpath, 'last.onnx')
        # print(f'\nSaving ONNX model to {model_script_path_onnx}')
        # # network.to_onnx(  # only old exporter, doesn't seem to work with pytorch lightning
        # #     file_path=model_script_path_onnx, input_sample=make_example_data())
        # from torch.onnx import dynamo_export  # new exporter supports control flow
        # dynamo_export(network, model_kwargs={'batch': make_example_data()}).save(model_script_path_onnx)

        model_script_path_libtorch = os.path.join(self.dirpath, 'last.pt')
        print(f'\nSaving TorchScript model to {model_script_path_libtorch}')
        make_dir_for_file(file=model_script_path_libtorch)
        network.to_torchscript(file_path=model_script_path_libtorch, method='trace', example_inputs=make_example_data())


class Cli(LightningCLI):
    def __init__(self, model_class, subclass_mode_model, datamodule_class, subclass_mode_data):
        print('{}: Starting {}'.format(get_now_str(), ' '.join(sys.argv)))
        sys.argv = self.handle_rec_subcommand(sys.argv)  # only call this with args from system command line
        super().__init__(
            model_class=model_class, subclass_mode_model=subclass_mode_model,
            datamodule_class=datamodule_class, subclass_mode_data=subclass_mode_data,
            save_config_kwargs={'overwrite': True})
        print('{}: Finished {}'.format(get_now_str(), ' '.join(sys.argv)))

    def cur_config(self) -> Namespace:
        return self.config[self.config.subcommand]

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # fundamentals
        parser.add_argument('--debug', type=bool, default=False,
                            help='set to True if you want debug outputs to validate the model')

    @abc.abstractmethod
    def handle_rec_subcommand(self, args: typing.List[str]) -> typing.List[str]:
        """
        Replace rec subcommand with predict and its default parameters before any argparse.
        Args:
            args: typing.List[str]

        Returns:
            new_args: typing.List[str]
        """
        pass

    # def before_fit(self):
    #     pass
    #
    # def after_fit(self):
    #     pass
    #
    # def before_predict(self):
    #     pass
    #
    # def after_predict(self):
    #     pass

    def before_instantiate_classes(self):
        import torch
        # torch.set_float32_matmul_precision('medium')  # PPSurf 50NN: 5.123h, ABC CD 0.012920511
        torch.set_float32_matmul_precision('high')  # PPSurf 50NN: xh, ABC CD y
        # torch.set_float32_matmul_precision('highest')  # PPSurf 50NN: xh, ABC CD y

        if bool(self.cur_config().debug):
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_DISTRIBUTED_DEBUG '] = '1'

            self.cur_config().trainer.detect_anomaly = True

    # def instantiate_classes(self):
    #     pass

    # def instantiate_trainer(self):
    #     pass

    # def parse_arguments(self, parser, args):
    #     pass

    # def setup_parser(self, add_subcommands, main_kwargs, subparser_kwargs):
    #     pass

    @staticmethod
    def subcommands() -> typing.Dict[str, typing.Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            'fit': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
            # 'validate': {'model', 'dataloaders', 'datamodule'}, # no val for this
            'test': {'model', 'dataloaders', 'datamodule'},
            'predict': {'model', 'dataloaders', 'datamodule'},
            # 'tune': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
        }
