from torch import Tensor, nn
from torchjd import TensorHierarchy
from torchjd.criterion import Criterion
from torchjd.module_wrapper import ModuleWrapper
from torchjd.transform import Grad, Store
from torchjd.tree import Leaf

from jde.loss_combiners import LossCombiner


class LossCombinationGDWrapper(ModuleWrapper):
    """
    `ModuleWrapper` that uses a `model` and a `criterion` to compute a 1-D loss tensor, and
    aggregates it into a scalar (0-D) tensor using a `loss_combiner`. It creates a
    `torchjd.TensorHierarchy` that can differentiate the combined loss with respect to all model
    parameters.
    """

    def __init__(self, model: nn.Module, criterion: Criterion, loss_combiner: LossCombiner):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.loss_combiner = loss_combiner

    def forward(self, x: Tensor, target: Tensor | list[Tensor]) -> TensorHierarchy:
        """
        Computes the output of the `model` on some input `x`. Uses the `criterion` on this output
        and on the `target` to get a loss vector. Combines this loss tensor into a loss scalar.
        Creates and returns a `torchjd.TensorHierarchy` that can differentiate the combined loss
        with respect to all model parameters.
        """

        output = self.model(x)
        loss_vector = self.criterion(output, target)
        loss_scalar = Leaf(self.loss_combiner(loss_vector))
        differentiation = Grad(loss_scalar, self.model.parameters())
        store = Store(self.model.parameters())
        transform = store << differentiation
        tensor = TensorHierarchy(loss_scalar, transform)

        return tensor
