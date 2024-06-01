from typing import Any

import torch
from torch import Tensor, nn

from torchjd.criterion._utils import _move_dim_back, _move_dim_to_front
from torchjd.criterion.base import Criterion


def _identity(output: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    return output, target


class SplitTensorCriterion(Criterion):
    """
    :class:`~torchjd.criterion.base.Criterion` that splits the output and the target
    :class:`Tensors <torch.Tensor>` into chunks and applies a loss function to each of those chunks.

    :param loss_function: The loss function to apply.
    :param dim: The dimension which separates the different chunks. It is the dimension along which
        the output and target :class:`Tensors <torch.Tensor>` are split.
    :param chunk_size: The number of elements along dimension ``dim`` that each chunk should
        contain. It should be a divisor of the length of dimension `dim` of the input tensors.
        Defaults to 1.
    :param pre_split_function: A function to apply to the (output, target) pair before splitting.
        Defaults to identity.

    .. admonition::
        Example

        Split a batch of 4 outputs and a batch of 4 targets along the batch dimension, to make
        chunks of size 2, then compute for each chunk the mean squared error between the output and
        the corresponding target.

        >>> import torch
        >>> from torch.nn import MSELoss
        >>> from torchjd.criterion import SplitTensorCriterion
        >>>
        >>> output = torch.tensor([0.0, 2.0, 5.0, 3.0])
        >>> target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>>
        >>> criterion = SplitTensorCriterion(MSELoss(), dim=0, chunk_size=2)
        >>> loss_vector = criterion(output, target)
        >>> loss_vector
        tensor([0.5000, 2.5000])

    .. admonition::
        Example

        Split a batch of classification outputs and a batch of classification targets along the
        class dimension. Then, compute the binary cross entropy between each output class and the
        corresponding target class.

        >>> import torch
        >>> from torch.nn import BCELoss
        >>> from torch.functional import F
        >>> from torchjd.criterion import SplitTensorCriterion
        >>>
        >>> N_CLASSES = 3
        >>> output = torch.tensor([[0.8, 0.1, 0.1], [0.0, 0.1, 0.9]])  # Class probabilities
        >>> target = torch.tensor([0, 2])  # Actual classes
        >>>
        >>> def encode_target(output, target):
        ...     processed_target = F.one_hot(target, num_classes=N_CLASSES).to(dtype=torch.float32)
        ...     return output, processed_target
        >>>
        >>> criterion = SplitTensorCriterion(
        ...     loss_function=BCELoss(),
        ...     dim=1,
        ...     pre_split_function=encode_target
        ... )
        >>> loss_vector = criterion(output, target)
        >>> loss_vector
        tensor([0.1116, 0.1054, 0.1054])
    """

    def __init__(
        self,
        loss_function: nn.Module,
        dim: int,
        chunk_size: int = 1,
        pre_split_function: callable = _identity,
    ):
        super().__init__()

        self.pre_split_function = pre_split_function
        self.loss_function = loss_function
        self.dim = dim
        self.chunk_size = chunk_size

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Processes ``output`` and ``target`` using ``pre_split_function``, then splits them into
        several chunks of size ``chunk_size``, along dimension ``dim``. Computes the value of the
        ``loss_function`` on each chunk, and returns the result as a 1-D loss
        :class:`~torch.Tensor`.

        :param output: The model's output.
        :param target: The value that we expect.

        .. note::
            This method should generally not be called directly. Instead, the Criterion instance
            should be called (like in the usage example), as this will take care of running
            potential hooks in addition to calling ``forward``.
        """

        output, target = self.pre_split_function(output, target)
        self._check_parameters(output, target)

        losses = []

        reshaped_output = _move_dim_to_front(output, self.dim)
        reshaped_target = _move_dim_to_front(target, self.dim)

        n_chunks = len(reshaped_output) // self.chunk_size

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            output_subtensor = _move_dim_back(reshaped_output[start:end], self.dim)
            target_subtensor = _move_dim_back(reshaped_target[start:end], self.dim)
            loss = self.loss_function(output_subtensor, target_subtensor)
            losses.append(loss)
        return torch.stack(losses)

    def _check_parameters(self, output: Any, target: Any) -> None:
        self._check_parameter(output, "output")
        self._check_parameter(target, "target")

        error_prefix = self._get_error_prefix()
        if output.shape[self.dim] != target.shape[self.dim]:
            raise ValueError(
                error_prefix + "parameters `output` and `target` should be of the same length along"
                f" dimension {self.dim} (the specified value of `dim`). Found `output.shape = "
                f"{output.shape}` and `target.shape = {target.shape}`."
            )

    def _check_parameter(self, param: Any, param_name: str) -> None:
        """
        Checks that after applying the ``pre_split_function``, a parameter is of the right type, dim
        and shape.
        """

        error_prefix = self._get_error_prefix()

        if not isinstance(param, Tensor):
            raise TypeError(
                error_prefix + f"parameter `{param_name}` should be of type `torch.Tensor`. Found "
                f"`type({param_name}) = {type(param)}."
            )
        if param.dim() <= self.dim:
            raise ValueError(
                error_prefix + f"parameter `{param_name}` should be of dimension > {self.dim} (the "
                f"specified value of `dim`). Found `{param_name}.dim() = {param.dim()}`."
            )
        if param.shape[self.dim] % self.chunk_size != 0:
            raise ValueError(
                error_prefix + f"parameter `{param_name}` should have the length of its dimension "
                f"{self.dim} (the specified value of `dim`) be a multiple of {self.chunk_size} (the"
                f" specified value of `chunk_size`). Found `{param_name}.shape = {param.shape}`."
            )

    def _get_error_prefix(
        self,
    ):
        if self.pre_split_function != _identity:
            error_prefix = "After applying `pre_split_function` to it, "
        else:
            error_prefix = ""
        return error_prefix

    def __str__(self) -> str:
        base_name = f"SplitTensor-{self.dim}"
        chunk_size_str = f"-{self.chunk_size} " if self.chunk_size != 1 else " "
        loss_str = str(self.loss_function)
        return base_name + chunk_size_str + loss_str
