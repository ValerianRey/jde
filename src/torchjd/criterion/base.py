from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class Criterion(nn.Module, ABC):
    """
    Abstract base class for all Criterion modules. It has the role of computing loss vectors.
    """

    @abstractmethod
    def forward(self, output: Any, target: Any) -> Tensor:
        """
        Abstract method responsible for the computation of the loss vector (1-D
        :class:`~torch.Tensor`) from an ``output`` and a ``target``.

        :param output: The model's output.
        :param target: The value that we expect.
        """

        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
