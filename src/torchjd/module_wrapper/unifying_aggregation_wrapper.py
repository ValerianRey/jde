from torch import nn

from torchjd.aggregation import Aggregator
from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.transform import Jac, Store, Subset, make_aggregation
from torchjd.transform.strategy import UnifyingStrategy
from torchjd.tree import Leaf


class UnifyingAggregationWrapper(ModuleWrapper):
    def __init__(
        self,
        model: nn.Module,
        aggregator: Aggregator,
        parallel_chunk_size: int | None,
    ):
        super().__init__()
        self.model = model
        self.aggregator = aggregator
        self.parallel_chunk_size = parallel_chunk_size

    def forward(self, input: TensorHierarchy) -> TensorHierarchy:
        output = self.model(*input.data)
        to_backpropagate = list(input.transform.required_keys)
        inputs = list(self.model.parameters()) + to_backpropagate

        # Transform that computes the required jacobians
        jac = Jac([output], inputs, chunk_size=self.parallel_chunk_size)

        # Transform that defines the aggregation of the jacobians into gradients
        aggregation = make_aggregation(UnifyingStrategy(self.aggregator, inputs))

        # Transform that selects the gradients to keep back-propagating
        backpropagate = Subset(to_backpropagate, inputs)

        # Transform that stores the gradients with respect to the model's parameters
        store = Store(self.model.parameters()) << Subset(self.model.parameters(), inputs)

        # Combine these transforms and compose them with the input's transform
        transform = (input.transform << backpropagate | store) << aggregation << jac

        return TensorHierarchy(Leaf(output), transform)
