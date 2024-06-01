from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from torchjd.tensor import TensorHierarchy


class ModuleWrapper(nn.Module, ABC):
    """
    TODO
    """

    @abstractmethod
    def forward(self, *inputs: Any) -> TensorHierarchy:
        """
        TODO
        """

        raise NotImplementedError
