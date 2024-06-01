from typing import Callable

from torch import Tensor, nn
from torchmetrics import MetricCollection

from jde.metrics import MultiBatchWrapper, MultiModeWrapper


def make_criterion_hook(
    loss_metrics: MultiModeWrapper,
    output_metrics: MultiModeWrapper,
    per_batch_loss_metrics: MultiBatchWrapper,
) -> Callable:
    def hook(_: nn.Module, inputs: tuple[Tensor, ...], loss_vector: Tensor) -> None:
        output = inputs[0]
        target = inputs[1]
        batch_size = output.shape[0]
        loss_metrics.update(loss_vector, batch_size)
        per_batch_loss_metrics.update(loss_vector, batch_size)
        output_metrics.update(output, target)

    return hook


def make_aggregator_hook(
    output_direction_metrics: MetricCollection,
    gradient_jacobian_metrics: MultiBatchWrapper,
) -> Callable:
    def hook(_: nn.Module, inputs: tuple[Tensor, ...], vector: Tensor) -> None:
        matrix = inputs[0]
        output_direction = matrix @ vector
        output_direction_metrics.update(output_direction)
        gradient_jacobian_metrics.update(vector, matrix)

    return hook


def make_weighting_hook(weight_metrics: MetricCollection) -> Callable:
    def hook(_: nn.Module, __: tuple[Tensor, ...], weights: Tensor) -> None:
        weight_metrics.update(weights)

    return hook
