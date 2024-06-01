from typing import Any

from torch import Tensor

from torchjd.criterion import Criterion
from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.transform import Diagonalize, Jac
from torchjd.tree import Tree


class CriterionWrapper(ModuleWrapper):
    """
    TODO
    """

    def __init__(
        self,
        criteria: Tree[Criterion],
        parallel_chunk_size: int | None,
    ):
        super().__init__()
        self.criteria = criteria
        self.parallel_chunk_size = parallel_chunk_size

    def forward(self, output: TensorHierarchy, targets: Tree[Any]) -> TensorHierarchy:
        """
        TODO

        :param output: The output of a model.
        :param targets: The targets with which the output should be compared.
        """

        def apply_criterion(triplet: tuple) -> Tensor:
            criterion = triplet[0]
            output_ = triplet[1]
            target = triplet[2]

            return criterion(output_, target)

        loss_tree = self.criteria.zip(output.data, targets).map(apply_criterion)

        # Transform that turns the gradients into jacobians
        diag = Diagonalize(loss_tree)

        # Transform that computes the required jacobians
        jac = Jac(loss_tree, output.data, chunk_size=self.parallel_chunk_size)

        # Combine these transforms and compose them with output's transform
        transform = output.transform << jac << diag

        return TensorHierarchy(loss_tree, transform)
