import torch

from torchjd import TensorHierarchy
from torchjd.module_wrapper import ModuleWrapper
from torchjd.transform import Store
from torchjd.tree import Leaf


class TensorBuilder(ModuleWrapper):
    def forward(self, tensor: torch.Tensor) -> TensorHierarchy:
        if tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad):
            to_store = [tensor]
        else:
            to_store = []

        return TensorHierarchy(Leaf(tensor), Store(to_store))
