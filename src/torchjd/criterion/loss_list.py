import torch
from torch import Tensor, nn

from torchjd.criterion.base import Criterion


class LossListCriterion(Criterion):
    """
    :class:`~torchjd.criterion.base.Criterion` that applies several loss functions to a single model
    output and target.

    :param loss_functions: The loss functions to apply.

    .. admonition::
        Example

        Compute the L1 and L2 loss functions (mean absolute error and mean squared error,
        respectively), on a given (``output``, ``target``) pair.

        >>> import torch
        >>> from torch.nn import L1Loss, MSELoss
        >>> from torchjd.criterion import LossListCriterion
        >>>
        >>> output = torch.tensor([0.0, 2.0, 5.0])
        >>> target = torch.tensor([1.0, 2.0, 3.0])
        >>>
        >>> criterion = LossListCriterion(loss_functions=[L1Loss(), MSELoss()])
        >>> loss_vector = criterion(output, target)
        >>> loss_vector
        tensor([1.0000, 1.6667])
    """

    def __init__(self, loss_functions: list[nn.Module]):
        super().__init__()
        self.loss_functions = nn.ModuleList(loss_functions)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes each loss function on the provided (``output``, ``target``) pair and returns the
        result as a 1-D loss :class:`~torch.Tensor`.

        :param output: The model's output.
        :param target: The value that we expect.

        .. note::
            This method should generally not be called directly. Instead, the Criterion instance
            should be called (like in the usage example), as this will take care of running
            potential hooks in addition to calling ``forward``.
        """

        losses = [loss_function(output, target) for loss_function in self.loss_functions]
        return torch.stack(losses)

    def __str__(self) -> str:
        loss_function_names = [str(loss_function) for loss_function in self.loss_functions]
        return "[" + ", ".join(loss_function_names) + "]"
