from abc import ABC, abstractmethod
from typing import Any

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torchjd.aggregation import Aggregator
from torchjd.criterion import Criterion
from torchjd.module_wrapper import FullJDWrapper, ModuleWrapper

from jde.architectures import ParameterizedModule
from jde.loss_combiners import LossCombiner
from jde.metrics import Metrics
from jde.module_wrappers import LossCombinationGDWrapper
from jde.printing import dict_to_str


class Solution(ABC):
    """
    Abstract base class for defining solutions to deep learning optimization problems, with methods
    for instantiating the objects used for training.

    :param drop_last_batch: Whether to drop the last batch if it contains fewer elements than the
        specified batch size.
    """

    header_color = None

    def __init__(
        self,
        criterion: Criterion,
        train_batch_size: int,
        evaluation_batch_size: int,
        metrics: Metrics,
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        architecture: type[ParameterizedModule],
        architecture_kwargs: dict | None,
        lr_scheduler_class: type[LRScheduler] | None,
        lr_scheduler_kwargs: dict | None,
        drop_last_batch: bool,
    ):
        if architecture_kwargs is None:
            architecture_kwargs = {}

        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}

        if lr_scheduler_class is None:
            # Constant learning rate by default
            lr_scheduler_class = StepLR
            lr_scheduler_kwargs = {"step_size": 1, "gamma": 1.0}

        self.criterion = criterion
        self.train_batch_size = train_batch_size
        self.evaluation_batch_size = evaluation_batch_size
        self.metrics = metrics
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.architecture = architecture
        self.architecture_kwargs = architecture_kwargs
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.drop_last_batch = drop_last_batch

    def _make_model(self) -> ParameterizedModule:
        return self.architecture(**self.architecture_kwargs)

    @abstractmethod
    def make_wrapper(self) -> ModuleWrapper:
        raise NotImplementedError

    def make_optimizer(self, model: Module) -> Optimizer:
        return self.optimizer_class(model.parameters(), **self.optimizer_kwargs)

    def make_lr_scheduler(self, optimizer: Optimizer):
        return self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def config(self) -> dict[str, Any]:
        optimizer_config = {
            f"optimizer_{key}": value for key, value in self.optimizer_kwargs.items()
        }
        architecture_config = {
            f"architecture_{key}": str(value) for key, value in self.architecture_kwargs.items()
        }
        lr_scheduler_config = {
            f"lr_scheduler_{key}": str(value) for key, value in self.lr_scheduler_kwargs.items()
        }

        return (
            {
                "solution": str(self),
                "criterion": str(self.criterion),
                "train_bs": self.train_batch_size,
                "eval_bs": self.evaluation_batch_size,
                "optimizer": self.optimizer_class.__name__,
                "architecture": self.architecture.__name__,
                "lr_scheduler_class": self.lr_scheduler_class.__name__,
                "drop_last_batch": self.drop_last_batch,
                "optimization_type": self.optimization_type,
            }
            | optimizer_config
            | architecture_config
            | lr_scheduler_config
        )

    @property
    @abstractmethod
    def optimization_type(self) -> str:
        raise NotImplementedError

    @property
    def _base_str(self) -> str:
        return f"{self.optimization_type} of {self.criterion} (BS={self.train_batch_size})"

    @property
    def _optimizer_str(self):
        return f"{self.optimizer_class.__name__} {dict_to_str(self.optimizer_kwargs)}"

    @property
    def _lr_scheduler_str(self):
        if self.lr_scheduler_class is StepLR and self.lr_scheduler_kwargs["gamma"] == 1.0:
            return "CLR"  # constant LR
        else:
            return f"{self.lr_scheduler_class.__name__} {dict_to_str(self.lr_scheduler_kwargs)}"


class GradientDescentSolution(Solution):
    header_color = "purple"

    def __init__(
        self,
        criterion: Criterion,
        train_batch_size: int,
        evaluation_batch_size: int,
        loss_combiner: LossCombiner,
        metrics: Metrics,
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        architecture: type[ParameterizedModule],
        architecture_kwargs: dict | None = None,
        lr_scheduler_class: type[LRScheduler] | None = None,
        lr_scheduler_kwargs: dict | None = None,
        drop_last_batch: bool = False,
    ):
        super().__init__(
            criterion=criterion,
            train_batch_size=train_batch_size,
            evaluation_batch_size=evaluation_batch_size,
            metrics=metrics,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            drop_last_batch=drop_last_batch,
        )
        self.loss_combiner = loss_combiner

    def make_wrapper(self) -> LossCombinationGDWrapper:
        model = self._make_model()
        return LossCombinationGDWrapper(
            model=model, criterion=self.criterion, loss_combiner=self.loss_combiner
        )

    def __str__(self) -> str:
        elements = [
            self._base_str,
            str(self.loss_combiner),
            self._optimizer_str,
            self._lr_scheduler_str,
        ]
        return " - ".join(elements)

    @property
    def config(self) -> dict[str, Any]:
        return super().config | {"loss_combiner": str(self.loss_combiner)}

    @property
    def optimization_type(self) -> str:
        return "GD"


class JDSolution(Solution, ABC):
    def __init__(
        self,
        criterion: Criterion,
        train_batch_size: int,
        evaluation_batch_size: int,
        aggregator: Aggregator,
        metrics: Metrics,
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        architecture: type[ParameterizedModule],
        architecture_kwargs: dict | None,
        lr_scheduler_class: type[LRScheduler] | None,
        lr_scheduler_kwargs: dict | None,
        drop_last_batch: bool,
        parallel_chunk_size: int | None,
    ):
        super().__init__(
            criterion=criterion,
            train_batch_size=train_batch_size,
            evaluation_batch_size=evaluation_batch_size,
            metrics=metrics,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            drop_last_batch=drop_last_batch,
        )
        self.aggregator = aggregator
        self.parallel_chunk_size = parallel_chunk_size

    def reset_aggregator(self):
        """
        Aggregators and Weightings defined in torchjd are supposed to be stateless. However,
        some of the aggregators defined in the literature are stateful, and thus rely on a reset
        method. To handle these cases, until we have a better solution, we can simply call the reset
        method if it exists, between each experiment.
        """

        if hasattr(self.aggregator, "reset"):
            self.aggregator.reset()
        if hasattr(self.aggregator, "weighting"):
            if hasattr(self.aggregator.weighting, "reset"):
                self.aggregator.weighting.reset()

    @property
    def config(self) -> dict[str, Any]:
        return super().config | {
            "aggregator": str(self.aggregator),
            "aggregator_repr": repr(self.aggregator),
            "parallel_chunk_size": str(self.parallel_chunk_size),
        }

    def __str__(self) -> str:
        elements = [
            self._base_str,
            str(self.aggregator),
            self._optimizer_str,
            self._lr_scheduler_str,
        ]
        return " - ".join(elements)


class FullJDSolution(JDSolution):
    header_color = "green"

    def __init__(
        self,
        criterion: Criterion,
        train_batch_size: int,
        evaluation_batch_size: int,
        aggregator: Aggregator,
        metrics: Metrics,
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        architecture: type[ParameterizedModule],
        architecture_kwargs: dict | None = None,
        lr_scheduler_class: type[LRScheduler] | None = None,
        lr_scheduler_kwargs: dict | None = None,
        drop_last_batch: bool = False,
        parallel_chunk_size: int | None = None,
    ):
        super().__init__(
            criterion=criterion,
            train_batch_size=train_batch_size,
            evaluation_batch_size=evaluation_batch_size,
            aggregator=aggregator,
            metrics=metrics,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            architecture=architecture,
            architecture_kwargs=architecture_kwargs,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            drop_last_batch=drop_last_batch,
            parallel_chunk_size=parallel_chunk_size,
        )

    def make_wrapper(self) -> FullJDWrapper:
        self.reset_aggregator()
        model = self._make_model()
        return FullJDWrapper(
            model=model,
            criterion=self.criterion,
            aggregator=self.aggregator,
            parallel_chunk_size=self.parallel_chunk_size,
        )

    @property
    def optimization_type(self) -> str:
        return "FJD"
