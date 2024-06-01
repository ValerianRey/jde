from copy import copy, deepcopy
from typing import Generator, Iterable

import numpy as np
import pandas as pd
from torch import nn
from torch.optim import SGD, Optimizer
from torchjd.criterion import LossListCriterion, SplitTensorCriterion
from torchmetrics import Accuracy, MetricCollection

from jde.aggregators import (
    KEY_TO_AGGREGATOR,
    gradient_jacobian_metrics,
    output_direction_metrics,
    weight_metrics,
)
from jde.architectures import ParameterizedModule
from jde.hooks import make_criterion_hook
from jde.loss_combiners import SqueezeCombiner
from jde.loss_functions import NamedModule
from jde.metrics import LossMetric, Metrics, MultiBatchWrapper, MultiModeWrapper
from jde.solutions import FullJDSolution, GradientDescentSolution, Solution


def generate_gd_solution(
    architecture: type[ParameterizedModule], lr: float
) -> Generator[Solution, None, None]:
    cce = NamedModule(nn.CrossEntropyLoss(), name="CCE")
    criterion = LossListCriterion(loss_functions=[cce])

    acc = Accuracy(task="multiclass", num_classes=architecture.N_CLASSES)
    output_metrics = MultiModeWrapper(MetricCollection({"Acc": acc}))
    loss_metrics = MultiModeWrapper(MetricCollection({cce.name: LossMetric()}))
    per_batch_loss_metrics = MultiBatchWrapper(deepcopy(loss_metrics))
    metrics = Metrics(
        output_metrics=output_metrics,
        loss_metrics=loss_metrics,
        per_batch_loss_metrics=per_batch_loss_metrics,
    )

    criterion_hook = make_criterion_hook(loss_metrics, output_metrics, per_batch_loss_metrics)
    criterion.register_forward_hook(criterion_hook)

    solution = GradientDescentSolution(
        criterion=criterion,
        train_batch_size=32,
        evaluation_batch_size=32,
        loss_combiner=SqueezeCombiner(),
        metrics=metrics,
        optimizer_class=SGD,
        optimizer_kwargs={"lr": lr},
        architecture=architecture,
        drop_last_batch=False,
    )

    yield solution


def generate_iwrm_solutions(
    architecture: type[ParameterizedModule],
    lr_df: pd.DataFrame,
    n_lr: int,
    batch_size: int,
    optimizer_class: type[Optimizer],
    optimizer_kwargs: dict,
) -> Generator[Solution, None, None]:
    cce = NamedModule(nn.CrossEntropyLoss(), name="CCE")
    ssjd_criterion = SplitTensorCriterion(loss_function=cce, dim=0)

    acc = Accuracy(task="multiclass", num_classes=architecture.N_CLASSES)
    output_metrics = MultiModeWrapper(MetricCollection({"Acc": acc}))
    loss_metrics = MultiModeWrapper(MetricCollection({cce.name: LossMetric()}))
    per_batch_loss_metrics = MultiBatchWrapper(deepcopy(loss_metrics))
    default_metrics = Metrics(
        output_metrics=output_metrics,
        loss_metrics=loss_metrics,
        per_batch_loss_metrics=per_batch_loss_metrics,
        output_direction_metrics=output_direction_metrics,
        vector_matrix_metrics=gradient_jacobian_metrics,
        weight_metrics=weight_metrics,
    )
    no_weights_metrics = Metrics(
        output_metrics=output_metrics,
        loss_metrics=loss_metrics,
        per_batch_loss_metrics=per_batch_loss_metrics,
        output_direction_metrics=output_direction_metrics,
        vector_matrix_metrics=gradient_jacobian_metrics,
    )

    criterion_hook = make_criterion_hook(loss_metrics, output_metrics, per_batch_loss_metrics)
    ssjd_criterion.register_forward_hook(criterion_hook)

    for _, (group_name, min_lr_exp, max_lr_exp) in lr_df.iterrows():
        aggregator = KEY_TO_AGGREGATOR[group_name]

        if hasattr(aggregator, "weighting"):
            metrics = default_metrics
        else:
            metrics = no_weights_metrics

        solution = FullJDSolution(
            criterion=ssjd_criterion,
            train_batch_size=batch_size,
            evaluation_batch_size=batch_size,
            aggregator=aggregator,
            metrics=metrics,
            optimizer_class=optimizer_class,
            optimizer_kwargs=copy(optimizer_kwargs),
            architecture=architecture,
            drop_last_batch=False,
        )

        learning_rates = np.geomspace(10**min_lr_exp, 10**max_lr_exp, n_lr)
        yield from _generator_from_learning_rates(solution, learning_rates)


def _generator_from_learning_rates(
    solution: Solution, learning_rates: Iterable[float]
) -> Generator[Solution, None, None]:
    for lr in learning_rates:
        new_optimizer_kwargs = copy(solution.optimizer_kwargs)
        new_optimizer_kwargs["lr"] = lr
        new_solution = copy(solution)
        new_solution.optimizer_kwargs = new_optimizer_kwargs
        yield new_solution
