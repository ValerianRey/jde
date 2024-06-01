from torch import nn


class NamedModule(nn.Module):
    """Wrapper around a nn.Module that adds a name field, and overrides __str__."""

    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def __str__(self) -> str:
        if self.name is None:
            return "Unnamed"
        else:
            return self.name
