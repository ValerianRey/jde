import torch

from torchjd.transform import EmptyTensorDict, Init, TensorDict, Transform
from torchjd.tree import Tree


class TensorHierarchy:
    """
    Class for representing some hierarchy of :class:`Tensors <torch.Tensor>` that can be
    differentiated using a :class:`~torchjd.transform.base.Transform`.

    :param data: The hierarchy of values.
    :param transform: The definition of the backward pass.
    """

    def __init__(self, data: Tree[torch.Tensor], transform: Transform[TensorDict, EmptyTensorDict]):
        self.data = data
        self.transform = transform

    def backward(self) -> None:
        """Applies the backward pass defined by the ``transform``."""

        initialized_transform = self.transform << Init(self.transform.required_keys)
        result = initialized_transform(EmptyTensorDict())

        if len(result) != 0:
            raise ValueError(
                "The backward pass is malformed, the transform is not supposed to return a "
                "non-empty dictionary"
            )

    def __str__(self) -> str:
        def to_str(tensor: torch.Tensor) -> str:
            # Extract the tensor string without autograd info (Eg: tensor([1., 2., 3.]))
            data_str = str(tensor.data)
            # Remove "tensor" and the parenthesis (Eg: [1., 2., 3.])
            result = f"{data_str[7:-1]}"
            return result

        str_tree = self.data.map(to_str)

        return "(" + str(str_tree) + f", transform={str(self.transform)})"
