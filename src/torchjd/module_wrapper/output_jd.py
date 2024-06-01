from torch import nn

from torchjd.aggregation import Aggregator
from torchjd.criterion import Criterion
from torchjd.module_wrapper.partial_jd import PartialJDWrapper


class OutputJDWrapper(PartialJDWrapper):
    """
    Special case of
    :class:`~torchjd.module_wrapper.partial_jd.PartialJDWrapper` in which the aggregation of the
    backward pass only considers the jacobian of the loss vector with respect to the output of the
    model.

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

        Compute the forward and the backward pass of an iteration of partial jacobian descent
        considering only the output of the model, on a simple regression model, using the mean
        squared error (MSE) and mean absolute error (MAE, also called L1).

        >>> import torch
        >>> from torch.nn import L1Loss, MSELoss, Sequential, Linear, ReLU
        >>>
        >>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
        >>> from torchjd.criterion import LossListCriterion
        >>> from torchjd.module_wrapper import OutputJDWrapper
        >>>
        >>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic
        >>>
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        >>> criterion = LossListCriterion([MSELoss(), L1Loss()])
        >>> W = UPGradWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>>
        >>> jd_wrapper = OutputJDWrapper(model, criterion, A)
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
        model: nn.Module,
        criterion: Criterion,
        aggregator: Aggregator,
        parallel_chunk_size: int | None = None,
    ):
        super().__init__(
            model_start=model,
            model_end=nn.Identity(),
            criterion=criterion,
            aggregator=aggregator,
            parallel_chunk_size=parallel_chunk_size,
        )
