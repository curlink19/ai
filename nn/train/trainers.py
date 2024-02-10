import torch
from typing import Optional

from nn.configs.train_configs import TrainConfig


class Trainer:
    def reset(self) -> None:
        """
        Backpropagation.
        """
        raise NotImplementedError

    def step(self, predicted, target) -> float:
        """
        Returns step loss.
        """
        raise NotImplementedError

    def get_loss(self, predicted, target) -> float:
        """
        Just returns loss.
        """
        raise NotImplementedError

    def end_epoch(self) -> None:
        """
        Update internal state after epoch.
        """
        raise NotImplementedError


class DefaultTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        config: TrainConfig,
        scheduler: Optional[torch.optim.lr_scheduler] = None,
    ):
        """
        Default trainer with the ability to accumulate batches.
        """

        self.optimizer = optimizer
        self.optimizer.zero_grad()

        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.steps_accumulated = config.batches_accumulated
        self.batch_size: int = config.batch_size
        self.current_step: int = 0

        if self.steps_accumulated == 1:
            assert loss_fn.reduction == "mean"
            self.factor = 1
        else:
            assert loss_fn.reduction == "sum"
            self.factor = 1 / (self.steps_accumulated * self.batch_size)

    def reset(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, predicted, target) -> float:
        self.current_step += 1

        loss = self.loss_fn(predicted, target) * self.factor
        loss.backward()

        if self.current_step % self.steps_accumulated == 0:
            self.reset()

        if self.steps_accumulated == 1:
            return loss.item()
        else:
            return loss.item() * self.steps_accumulated

    @torch.no_grad()
    def get_loss(self, predicted, target) -> float:
        loss = self.loss_fn(predicted, target)

        if self.steps_accumulated == 1:
            return loss.item()
        else:
            return loss.item() / self.batch_size

    def end_epoch(self) -> None:
        self.current_step = 0

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad()
