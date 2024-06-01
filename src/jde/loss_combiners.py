from abc import ABC, abstractmethod

from torch import Tensor, nn


class LossCombiner(nn.Module, ABC):
    """
    A loss combiner is an object that is responsible for combining a 1-D loss tensor into a 0-D
    loss tensor. It should typically be used to get a single scalar loss and perform traditional
    (gradient descent) optimization.
    """

    @abstractmethod
    def forward(self, loss_tensor: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class SqueezeCombiner(LossCombiner):
    """
    This dummy combiner takes a 1-D loss tensor of size 1 and squeezes it into a 0-D loss tensor.
    """

    def forward(self, loss_tensor: Tensor) -> Tensor:
        return loss_tensor.squeeze()

    def __str__(self) -> str:
        return "squeeze"
