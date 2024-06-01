from torch import nn

from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.transform import Grad, Store, Subset
from torchjd.tree import Leaf


class GradWrapper(ModuleWrapper):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    def forward(self, input: TensorHierarchy) -> TensorHierarchy:
        output = self.model(*input.data)
        to_backpropagate = list(input.transform.required_keys)
        inputs = list(self.model.parameters()) + to_backpropagate

        # Transform that computes the required gradients
        grad = Grad([output], inputs)

        # Transform that selects the gradients to keep back-propagating
        backpropagate = Subset(to_backpropagate, inputs)

        # Transform that stores the gradients with respect to the model's parameters
        store = Store(self.model.parameters()) << Subset(self.model.parameters(), inputs)

        # Combine these transforms and compose them with the input's transform
        transform = (input.transform << backpropagate | store) << grad

        return TensorHierarchy(Leaf(output), transform)
