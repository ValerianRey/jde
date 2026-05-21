from torch import Tensor, nn
from torchjd.aggregation import Aggregator
from torchjd.autogram import Engine
from torchjd.autojac import backward as jac_backward
from torchjd.autojac import jac_to_grad

from jde.criteria import Criterion
from jde.loss_combiners import LossCombiner


class _JDResult:
    """Holds the loss vector and backward context for a JD forward pass."""

    def __init__(
        self,
        losses: Tensor,
        params: list,
        aggregator: Aggregator,
        parallel_chunk_size: int | None,
    ):
        self._losses = losses
        self._params = params
        self._aggregator = aggregator
        self._parallel_chunk_size = parallel_chunk_size

    def backward(self) -> None:
        jac_backward(
            self._losses,
            inputs=self._params,
            parallel_chunk_size=self._parallel_chunk_size,
        )
        jac_to_grad(iter(self._params), self._aggregator)


class _GramianResult:
    """Holds the loss vector and backward context for a gramian-based JD forward pass."""

    def __init__(self, losses: Tensor, engine: Engine, gramian_weighting: nn.Module):
        self._losses = losses
        self._engine = engine
        self._gramian_weighting = gramian_weighting

    def backward(self) -> None:
        gramian = self._engine.compute_gramian(self._losses)
        weights = self._gramian_weighting(gramian)
        self._losses.backward(weights)


class LossCombinationGDWrapper(nn.Module):
    """
    Module that runs a model and criterion, combines the resulting loss vector into a scalar, and
    returns it. Call `.backward()` on the result to differentiate via standard gradient descent.
    """

    def __init__(self, model: nn.Module, criterion: Criterion, loss_combiner: LossCombiner):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.loss_combiner = loss_combiner

    def forward(self, x: Tensor, target: Tensor | list[Tensor]) -> Tensor:
        output = self.model(x)
        loss_vector = self.criterion(output, target)
        return self.loss_combiner(loss_vector)


class FullJDWrapper(nn.Module):
    """
    Module that runs a model and criterion to produce a loss vector, and returns a `_JDResult`
    whose `.backward()` computes the Jacobian and aggregates it into `.grad` via torchjd.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        aggregator: Aggregator,
        parallel_chunk_size: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.aggregator = aggregator
        self.parallel_chunk_size = parallel_chunk_size

    def forward(self, x: Tensor, target: Tensor | list[Tensor]) -> _JDResult:
        output = self.model(x)
        loss_vector = self.criterion(output, target)
        params = list(self.model.parameters())
        return _JDResult(loss_vector, params, self.aggregator, self.parallel_chunk_size)


class GramianJDWrapper(nn.Module):
    """
    Module that runs a model and criterion to produce a loss vector, and returns a `_GramianResult`
    whose `.backward()` computes the Gramian via autogram's Engine and uses a GramianWeighting to
    produce per-loss weights for a standard weighted backward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        gramian_weighting: nn.Module,
        batch_dim: int = 0,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.gramian_weighting = gramian_weighting
        self._engine = Engine(model, batch_dim=batch_dim)

    def forward(self, x: Tensor, target: Tensor | list[Tensor]) -> _GramianResult:
        output = self.model(x)
        loss_vector = self.criterion(output, target)
        return _GramianResult(loss_vector, self._engine, self.gramian_weighting)
