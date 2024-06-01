from typing import Any

from torch import nn

from torchjd.aggregation import Aggregator
from torchjd.criterion import Criterion
from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.module_wrapper.criterion_wrapper import CriterionWrapper
from torchjd.module_wrapper.tensor_builder import TensorBuilder
from torchjd.module_wrapper.unifying_aggregation_wrapper import UnifyingAggregationWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.tree import Leaf


class FullJDWrapper(ModuleWrapper):
    """
    :class:`~torchjd.module_wrapper.base.ModuleWrapper` that enables full jacobian descent.

    It creates tensors whose backward pass can differentiate several losses with respect to all
    parameters of the model and aggregate the obtained jacobians.

    :param model: The model to apply to the input.
    :param criterion: The object responsible for computing the loss vector from the output of the
        model and the expected target.
    :param aggregator: The object responsible for aggregating the jacobian matrices.
    :param parallel_chunk_size: The number of losses to differentiate simultaneously in the
        backward pass. If set to ``None``, all losses will be differentiated in parallel in one go.
        If set to `1`, all losses will be differentiated sequentially. A larger value results in
        faster differentiation, but also higher memory usage. Defaults to ``None``.

    .. admonition::
        Example

        Compute the forward and the backward pass of an iteration of full jacobian descent on a
        simple regression model, using the mean squared error (MSE) and mean absolute error (MAE,
        also called L1).

        >>> import torch
        >>> from torch.nn import L1Loss, MSELoss, Sequential, Linear, ReLU
        >>>
        >>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
        >>> from torchjd.criterion import LossListCriterion
        >>> from torchjd.module_wrapper import FullJDWrapper
        >>>
        >>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic
        >>>
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        >>> criterion = LossListCriterion([MSELoss(), L1Loss()])
        >>> W = UPGradWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>>
        >>> jd_wrapper = FullJDWrapper(model, criterion, A)
        >>>
        >>> model_input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
        >>> target = model_input.sum(dim=1, keepdim=True)  # Batch of 16 targets
        >>>
        >>> losses = jd_wrapper(model_input, target)  # forward pass

        Note that ``losses`` is a :class:`~torchjd.tensor.tensor_hierarchy.TensorHierarchy`
        containing all the losses and equipped with a Jacobian descent backward pass.

        >>> losses.backward()  # backward pass

        The ``.grad`` field of each parameter of the model is now populated.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        aggregator: Aggregator,
        parallel_chunk_size: int | None = None,
    ):
        super().__init__()
        self.tensor_builder = TensorBuilder()
        self.model = UnifyingAggregationWrapper(model, aggregator, parallel_chunk_size)
        self.criterion = CriterionWrapper(Leaf(criterion), parallel_chunk_size)

    def forward(self, input: Any, target: Any) -> TensorHierarchy:
        """
        Computes the output of the ``model`` on some ``input``. Uses the ``criterion`` on this
        output and on the ``target`` to get a loss vector.

        Creates and returns a tensor that defines the following backward pass:

        - Differentiate all elements of the loss vector with respect to all parameters of the model.
        - Aggregate the obtained jacobians.

        :param input: The input to give to the model.
        :param target: The target with which the model output should be compared.
        """

        input_tensor = self.tensor_builder(input)
        output = self.model(input_tensor)
        loss_vector = self.criterion(output, Leaf(target))

        return loss_vector
