from typing import Any

from torch import nn

from torchjd.aggregation import Aggregator
from torchjd.criterion import Criterion
from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.module_wrapper.criterion_wrapper import CriterionWrapper
from torchjd.module_wrapper.grad_wrapper import GradWrapper
from torchjd.module_wrapper.tensor_builder import TensorBuilder
from torchjd.module_wrapper.unifying_aggregation_wrapper import UnifyingAggregationWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.tree import Leaf


class PartialJDWrapper(ModuleWrapper):
    """
    :class:`~torchjd.module_wrapper.base.ModuleWrapper` that enables partial jacobian descent.

    It creates tensors whose backward pass can differentiate several losses with respect to the
    parameters of some layers at the end of the model, aggregate the obtained jacobians, and apply
    chain rule to compute gradients with respect to the parameters of the rest of the model.

    .. note::
        The differentiation with respect to the parameters of ``model_end`` is performed once per
        element in the loss vector. The differentiation with respect to the parameters of
        ``model_start`` is performed only once.

    .. hint::
        This is a compromise between full jacobian descent, as provided by
        :class:`~torchjd.module_wrapper.full_jd.FullJDWrapper`, and the usual gradient descent. This
        translates into a compromise between computation time of the backward pass and the amount of
        parameters considered during the aggregation.

    :param model_start: The layers at the beginning of the model.
    :param model_end: The layers at the end of the model. The jacobians relative to them are
        considered in the aggregation.
    :param criterion: The object responsible for computing the loss vector from the output of the
        model and the expected target.
    :param aggregator: The object responsible for aggregating the jacobian matrices.
    :param parallel_chunk_size: The number of losses to differentiate simultaneously in the
        backward pass. If set to ``None``, all losses will be differentiated in parallel in one go.
        If set to `1`, all losses will be differentiated sequentially. A larger value results in
        faster differentiation, but also higher memory usage. Defaults to ``None``.

    .. admonition::
        Example

        Compute the forward and the backward pass of an iteration of partial jacobian descent on a
        simple regression model, using the mean squared error (MSE) and mean absolute error (MAE,
        also called L1).

        >>> import torch
        >>> from torch.nn import L1Loss, MSELoss, Sequential, Linear, ReLU
        >>>
        >>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
        >>> from torchjd.criterion import LossListCriterion
        >>> from torchjd.module_wrapper import PartialJDWrapper
        >>>
        >>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic
        >>>
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        >>> criterion = LossListCriterion([MSELoss(), L1Loss()])
        >>> W = UPGradWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>>
        >>> model_start = model[:2]
        >>> model_end = model[2:]
        >>> jd_wrapper = PartialJDWrapper(model_start, model_end, criterion, A)
        >>>
        >>> model_input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
        >>> target = model_input.sum(dim=1, keepdim=True)  # Batch of 16 targets
        >>>
        >>> losses = jd_wrapper(model_input, target)  # forward pass

        Note that ``losses`` is a :class:`~torchjd.TensorHierarchy` containing all the losses and a
        :class:`~torchjd.transform.base.Transform` defining the backward pass.

        >>> losses.backward()  # backward pass

        The ``.grad`` field of each parameter of the model is now populated.
    """

    def __init__(
        self,
        model_start: nn.Module,
        model_end: nn.Module,
        criterion: Criterion,
        aggregator: Aggregator,
        parallel_chunk_size: int | None = None,
    ):
        super().__init__()
        self.tensor_builder = TensorBuilder()
        self.model_start = GradWrapper(model_start)
        self.model_end = UnifyingAggregationWrapper(model_end, aggregator, parallel_chunk_size)
        self.criterion = CriterionWrapper(Leaf(criterion), parallel_chunk_size)

    def forward(self, input: Any, target: Any) -> TensorHierarchy:
        """
        Computes the intermediate activation tensor using ``model_start``, then computes the final
        output using ``model_end`` on this intermediate activation. Uses the ``criterion`` on this
        output and on the ``target`` to get a loss vector.

        Creates and returns a tensor that defines the following backward pass:

        - Differentiate all elements of the loss vector with respect to the parameters of
          ``model_end`` and with respect to the intermediate activation.
        - Aggregate the obtained jacobians.
        - Apply chain rule, with the obtained gradient with respect to the intermediate activation,
          in order to differentiate with respect to the parameters of ``model_start``.

        :param input: The input to give to the model.
        :param target: The target with which the model output should be compared.
        """

        input_tensor = self.tensor_builder(input)
        intermediate_activation = self.model_start(input_tensor)
        output = self.model_end(intermediate_activation)
        loss_vector = self.criterion(output, Leaf(target))

        return loss_vector
