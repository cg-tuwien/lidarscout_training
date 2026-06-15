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

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_end(trainer, pl_module)

        from source.base.fs import make_dir_for_file

        # Force PyTorch to use deterministic kernels that support dynamic batch sizes
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        network = trainer.lightning_module
        network.eval()

        # does not work with LightningModule forward() having *args and **kwargs
        # net_ts = network.to_torchscript(model_script_path, method='script')

        # save with tracing, need a training sample that will be traced through the forward pass
        def make_example_data():
            import torch
            
            # Trace with a batch size of 1 to prevent hardcoded multi-batch kernel optimizations
            bs = 1
        
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

            for m in network.input_methods:
                example_inputs[f'patch_hm_{m}'] = torch.rand(bs, 1, res, res)
                example_inputs[f'patch_rgb_{m}'] = torch.rand(bs, 3, res, res)
            example_inputs[f'patch_hm_mask'] = torch.rand(bs, 1, res, res)
            return example_inputs

        # model_script_path_onnx = os.path.join(self.dirpath, 'last.onnx')
        # print(f'\nSaving ONNX model to {model_script_path_onnx}')
        # # network.to_onnx(  # only old exporter, doesn't seem to work with pytorch lightning
        # #     file_path=model_script_path_onnx, input_sample=make_example_data())
        # from torch.onnx import dynamo_export  # new exporter supports control flow
        # dynamo_export(network, model_kwargs={'batch': make_example_data()}).save(model_script_path_onnx)

        # trace and save model for libtorch
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
        parser.add_argument('--refresh_cache', type=bool, default=False,
                            help='set to True to rebuild img_cache before fit')
        
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        import inspect
        
        # =================================================================
        # GAN Multi-Optimizer Interception
        # =================================================================
        if getattr(lightning_module, 'automatic_optimization', True) is False and hasattr(lightning_module, 'discriminator'):
            
            # 1. Clone the Optimizer Settings
            OptClass = type(optimizer)
            opt_defaults = optimizer.defaults.copy()

            # Dynamically filter defaults to only include valid __init__ arguments
            valid_kwargs = inspect.signature(OptClass.__init__).parameters.keys()
            filtered_defaults = {k: v for k, v in opt_defaults.items() if k in valid_kwargs}

            # Spawn two new optimizers using ONLY the valid config arguments
            opt_g = OptClass(lightning_module.regressor.parameters(), **filtered_defaults)
            opt_d = OptClass(lightning_module.discriminator.parameters(), **filtered_defaults)

            if lr_scheduler is None:
                return [opt_g, opt_d], []

            # 2. Clone the Scheduler Settings
            SchedulerClass = type(lr_scheduler)
            
            from torch.optim.lr_scheduler import MultiStepLR, StepLR
            if isinstance(lr_scheduler, MultiStepLR):
                # MultiStepLR internally converts milestones to a Counter object.
                # .elements() unpacks the Counter back into the raw list defined in YAML.
                milestones = list(lr_scheduler.milestones.elements())
                gamma = lr_scheduler.gamma
                
                sch_g = SchedulerClass(opt_g, milestones=milestones, gamma=gamma)
                sch_d = SchedulerClass(opt_d, milestones=milestones, gamma=gamma)
                
                return [opt_g, opt_d], [sch_g, sch_d]
                
            elif isinstance(lr_scheduler, StepLR):
                sch_g = SchedulerClass(opt_g, step_size=lr_scheduler.step_size, gamma=lr_scheduler.gamma)
                sch_d = SchedulerClass(opt_d, step_size=lr_scheduler.step_size, gamma=lr_scheduler.gamma)
                
                return [opt_g, opt_d], [sch_g, sch_d]
            else:
                raise NotImplementedError(f"Scheduler cloning for {SchedulerClass.__name__} is not implemented for GANs.")

        # =================================================================
        # Standard Single-Optimizer Logic (For non-GAN CNNs)
        # =================================================================
        if lr_scheduler is None:
            return optimizer

        # ReduceLROnPlateau needs special treatment
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            monitor_name = getattr(lr_scheduler, 'monitor', None)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': monitor_name},
            }
            
        return [optimizer], [lr_scheduler]

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
        
        # this is deprecated in pytorch, update when pytorch-lightning supports it:
        # torch.backends.cuda.matmul.fp32_precision = "tf32"
        # torch.backends.cudnn.conv.fp32_precision = "tf32"

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

    def before_fit(self):
        datamodule = self.datamodule
        required_attrs = [
            'pts_to_img_methods', 'rgb_to_img_methods',
            'train_set', 'val_set', 'in_file',
            'hm_interp_size', 'hm_size', 'context_radius_factor',
            'meters_per_pixel', 'dataset_step',
        ]
        if not all(hasattr(datamodule, attr) for attr in required_attrs):
            raise ValueError(f'Missing required datamodule attributes for img_cache precompute, expected {required_attrs} but got {datamodule.__dict__.keys()}')

        from source.dataloaders.img_cache_precompute import precompute_img_cache_for_fit

        refresh_cache = bool(getattr(self.cur_config(), 'refresh_cache', False))
        precompute_img_cache_for_fit(
            in_file=datamodule.in_file,
            train_set=datamodule.train_set,
            val_set=datamodule.val_set,
            hm_interp_size=datamodule.hm_interp_size,
            hm_size=datamodule.hm_size,
            context_radius_factor=datamodule.context_radius_factor,
            meters_per_pixel=datamodule.meters_per_pixel,
            dataset_step=datamodule.dataset_step,
            pts_to_img_methods=datamodule.pts_to_img_methods,
            rgb_to_img_methods=datamodule.rgb_to_img_methods,
            refresh_cache=refresh_cache,
        )
