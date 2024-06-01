from copy import deepcopy
from typing import Any

import torch
import wandb
from torch import Tensor
from torchmetrics import MeanMetric, Metric, MetricCollection

from jde.settings import DEVICE


class LossMetric(MeanMetric):
    """MeanMetric specifically made for something that we seek to minimize."""

    higher_is_better = False


class MultiModeWrapper(Metric):
    """
    Metric wrapper that allows to track metrics in several modes and to alternate between those
    modes.
    """

    MODES = ["training", "train_eval", "test_eval"]

    def __init__(self, metric: Metric | MetricCollection):
        super().__init__()

        self.metrics = {mode: deepcopy(metric).to(DEVICE) for mode in self.MODES}
        self._mode = self.MODES[0]

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in self.MODES:
            raise KeyError(value)

        self._mode = value

    @property
    def metric(self) -> Metric | MetricCollection:
        return self.metrics[self.mode]

    def update(self, *_: Any, **__: Any) -> None:
        self.metric.update(*_, **__)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        super().reset()


class CosineSimilarityToMatrixMean(Metric):
    """
    Metric that computes the cosine similarity between a vector and the average of the rows of a
    matrix.

    .. note::
        This can be useful to tell how far the result of a jacobian matrix aggregation is from the
        mean of this matrix (in terms of direction), which is equivalent to what gradient descent
        would give.
    """

    def __init__(self):
        super().__init__()
        self.add_state("similarities", default=[], dist_reduce_fx="mean")

    def update(self, vector: Tensor, matrix: Tensor) -> None:
        mean = matrix.mean(dim=0)
        similarity = torch.cosine_similarity(vector, mean, dim=0)
        self.similarities.append(similarity)

    def compute(self) -> Tensor:
        return torch.stack(self.similarities).mean()


class MultiBatchWrapper(MultiModeWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("metric_values", default=[], dist_reduce_fx="cat")

    def update(self, *_: Any, **__: Any) -> None:
        if self.mode == "training":
            self.metric_values.append(self.metric.forward(*_, **__))

    def compute(self) -> list:
        if self.mode == "training":
            return self.metric_values
        else:
            return []


class Metrics:
    def __init__(
        self,
        output_metrics: MultiModeWrapper,
        loss_metrics: MultiModeWrapper,
        per_batch_loss_metrics: MultiBatchWrapper,
        output_direction_metrics: MetricCollection | None = None,
        vector_matrix_metrics: MultiBatchWrapper | None = None,
        weight_metrics: MetricCollection | None = None,
    ):
        self.output_metrics = output_metrics
        self.loss_metrics = loss_metrics
        self.per_batch_loss_metrics = per_batch_loss_metrics
        self.output_direction_metrics = output_direction_metrics
        self.vector_matrix_metrics = vector_matrix_metrics
        self.weight_metrics = weight_metrics

        if not (output_metrics.mode == loss_metrics.mode == per_batch_loss_metrics.mode):
            raise ValueError("All modes should be the same.")

        self._mode = output_metrics.mode

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value
        self.output_metrics.mode = self._mode
        self.loss_metrics.mode = self._mode
        self.per_batch_loss_metrics.mode = self._mode

    # Small explanation about how logging works. First of all, be warned, it's extremely messy.
    # Due to the way wandb handles the step, we have to make sure to commit as sparingly as
    # possible. Each commit increases the step, and all metrics that were not provided during
    # the corresponding log will be replaced by NaN in their memory. This dramatically impacts
    # all performance related to wandb (UI on their website, download speed from API, ...). It will
    # also create bugs, like not being able to call the "run.history" method with several keys.

    # Unfortunately, it is not possible to make only 1 commit per epoch, because some of the metrics
    # are per-batch. The work-around solution is thus to make 1 commit per batch with all per-batch
    # metrics in it + 1 commit per epoch with all per-epoch metrics in it, including training,
    # train-eval and test metrics. This will create a lot of NaN values, because each per-batch
    # commit will include 1 NaN value for each per-epoch metric, so it is best to keep the number
    # of per-epoch metrics minimal (basically, they cost as much memory and compute as per-batch
    # metrics, because of how wandb works internally).

    def log_and_commit_per_batch_metrics(self, epoch: int, training: bool) -> None:
        if training:
            multiloss_metric_dicts = self.per_batch_loss_metrics.compute()

            if self.vector_matrix_metrics is not None:
                vector_matrix_metric_dicts = self.vector_matrix_metrics.compute()
            else:
                vector_matrix_metric_dicts = None

            # Only works if all epochs have the same number of batches
            n_batches_per_epoch = len(multiloss_metric_dicts)

            for batch_id in range(n_batches_per_epoch):
                current_batch = n_batches_per_epoch * epoch + batch_id
                to_log = {"batch": current_batch}
                empty_log = True

                for metric_name, metric_value in multiloss_metric_dicts[batch_id].items():
                    tag = f"losses/{metric_name}/training (per-batch)"
                    to_log[tag] = metric_value
                    empty_log = False

                if vector_matrix_metric_dicts is not None:
                    for metric_name, metric_value in vector_matrix_metric_dicts[batch_id].items():
                        tag = f"gradient-jacobian/{metric_name} (per-batch)"
                        to_log[tag] = metric_value
                        empty_log = False

                if not empty_log:
                    wandb.log(to_log, commit=True)

    def log_per_epoch_metrics(self, epoch: int, training: bool, commit: bool) -> None:
        # commit should be True only once per epoch
        self.log_output_metrics(epoch)
        self.log_multiloss(epoch)

        if training and self.output_direction_metrics is not None:
            self.log_output_direction_metrics(epoch)

        if training and self.weight_metrics is not None:
            self.log_weight_metrics(epoch)

        if commit:
            wandb.log({}, commit=True)

    def log_multiloss(self, epoch: int) -> None:
        multiloss_metric_dict = self.loss_metrics.compute()
        for metric_name, metric_value in multiloss_metric_dict.items():
            tag = f"losses/{metric_name}/{self.mode}"
            wandb.log({tag: metric_value, "epoch": epoch}, commit=False)

    def log_output_metrics(self, epoch: int) -> None:
        output_metric_dict = self.output_metrics.compute()
        for metric_name, metric_value in output_metric_dict.items():
            tag = f"output/{metric_name}/{self.mode}"
            wandb.log({tag: metric_value, "epoch": epoch}, commit=False)

    def log_output_direction_metrics(self, epoch: int) -> None:
        metric_dict = self.output_direction_metrics.compute()
        for metric_name, metric_value in metric_dict.items():
            tag = f"output direction/{metric_name}"
            wandb.log({tag: metric_value, "epoch": epoch}, commit=False)

    def log_weight_metrics(self, epoch: int) -> None:
        metric_dict = self.weight_metrics.compute()
        for metric_name, metric_value in metric_dict.items():
            tag = f"weight/{metric_name}"
            wandb.log({tag: metric_value, "epoch": epoch}, commit=False)

    def reset_all(self):
        self.output_metrics.reset()
        self.loss_metrics.reset()
        self.per_batch_loss_metrics.reset()

        if self.output_direction_metrics is not None:
            self.output_direction_metrics.reset()

        if self.vector_matrix_metrics is not None:
            self.vector_matrix_metrics.reset()

        if self.weight_metrics is not None:
            self.weight_metrics.reset()
