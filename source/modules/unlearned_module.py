from typing import Any, Optional, Callable, Union

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler


class UnlearnedModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return None

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        pass

    def manual_backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        pass

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        pass

    def common_step(self, batch, step: str):
        raise NotImplementedError('no training needed')

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('no training needed')

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError('no training needed')
