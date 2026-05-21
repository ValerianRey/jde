from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn


class Criterion(nn.Module, ABC):
    @abstractmethod
    def forward(self, output: Any, target: Any) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


def _move_dim_to_front(t: Tensor, dim: int) -> Tensor:
    axes_order = [dim] + list(range(dim)) + list(range(dim + 1, t.dim()))
    return t.permute(axes_order)


def _move_dim_back(t: Tensor, dim: int) -> Tensor:
    axes_order = list(range(t.dim()))[1:]
    axes_order.insert(dim, 0)
    return t.permute(axes_order)


class LossListCriterion(Criterion):
    def __init__(self, loss_functions: list[nn.Module]):
        super().__init__()
        self.loss_functions = nn.ModuleList(loss_functions)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        losses = [loss_function(output, target) for loss_function in self.loss_functions]
        return torch.stack(losses)

    def __str__(self) -> str:
        loss_function_names = [str(loss_function) for loss_function in self.loss_functions]
        return "[" + ", ".join(loss_function_names) + "]"


class SplitTensorCriterion(Criterion):
    def __init__(
        self,
        loss_function: nn.Module,
        dim: int,
        chunk_size: int = 1,
        pre_split_function: Callable | None = None,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.dim = dim
        self.chunk_size = chunk_size
        self.pre_split_function = pre_split_function or _identity

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        output, target = self.pre_split_function(output, target)

        reshaped_output = _move_dim_to_front(output, self.dim)
        reshaped_target = _move_dim_to_front(target, self.dim)

        n_chunks = len(reshaped_output) // self.chunk_size
        losses = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            output_chunk = _move_dim_back(reshaped_output[start:end], self.dim)
            target_chunk = _move_dim_back(reshaped_target[start:end], self.dim)
            losses.append(self.loss_function(output_chunk, target_chunk))
        return torch.stack(losses)

    def __str__(self) -> str:
        base_name = f"SplitTensor-{self.dim}"
        chunk_size_str = f"-{self.chunk_size} " if self.chunk_size != 1 else " "
        return base_name + chunk_size_str + str(self.loss_function)


def _identity(output: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    return output, target
