from typing import Any

from torch import nn

from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.module_wrapper.grad_wrapper import GradWrapper
from torchjd.module_wrapper.tensor_builder import TensorBuilder
from torchjd.tensor import TensorHierarchy
from torchjd.transform import Grad
from torchjd.tree import Leaf


class GDWrapper(ModuleWrapper):
    """
    :class:`~torchjd.module_wrapper.base.ModuleWrapper` that enables gradient descent.

    It creates tensors whose backward pass simply differentiates the loss scalar with respect to all
    parameters of the model.

    :param model: The model to apply to the input.
    :param loss_function: The object responsible for computing the loss scalar from the output of
        the model and the expected target.

    .. admonition::
        Example

        Compute the forward and the backward pass of an iteration of gradient descent on a simple
        regression model, using the mean squared error (MSE).

        >>> import torch
        >>> from torch.nn import MSELoss, Sequential, Linear, ReLU
        >>>
        >>> from torchjd.module_wrapper import GDWrapper
        >>>
        >>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic
        >>>
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        >>> loss_function = MSELoss()
        >>>
        >>> gd_wrapper = GDWrapper(model, loss_function)
        >>>
        >>> model_input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
        >>> target = model_input.sum(dim=1, keepdim=True)  # Batch of 16 targets
        >>>
        >>> loss = gd_wrapper(model_input, target)  # forward pass

        Note that ``loss`` is a :class:`~torchjd.TensorHierarchy` containing the loss tensor and a
        :class:`~torchjd.transform.base.Transform` defining the backward pass.

        >>> loss.backward()  # backward pass

        The ``.grad`` field of each parameter of the model is now populated.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
    ):
        super().__init__()
        self.tensor_builder = TensorBuilder()
        self.model = GradWrapper(model)
        self.loss_function = loss_function

    def forward(self, input: Any, target: Any) -> TensorHierarchy:
        """
        Computes the output of the ``model`` on some ``input``. Uses the ``loss_function`` on this
        output and on the ``target`` to get a loss scalar.

        Creates and returns a tensor that defines the following backward pass:

        - Differentiate the loss with respect to all parameters of the model.

        :param input: The input to give to the model.
        :param target: The target with which the model output should be compared.
        """

        input_tensor = self.tensor_builder(input)
        output = self.model(input_tensor)
        output_scalar = output.data.value
        loss_scalar = self.loss_function(output_scalar, target)

        # Transform that computes the required gradients
        grad = Grad([loss_scalar], [output_scalar])

        # Combine these transforms and compose them with output's transform
        transform = output.transform << grad

        return TensorHierarchy(Leaf(loss_scalar), transform)
