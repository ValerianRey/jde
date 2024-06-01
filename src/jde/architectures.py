from torch import Tensor, nn

from jde.settings import DEVICE, FLOAT_DTYPE


class ParameterizedModule(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()


class ClassificationModel:
    N_CLASSES = None


class MnistModel(ParameterizedModule, ClassificationModel):
    N_CLASSES = 10

    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(3),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ELU(),
            nn.Linear(128, self.N_CLASSES),
        ).to(DEVICE, dtype=FLOAT_DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Cifar10Model(ParameterizedModule, ClassificationModel):
    N_CLASSES = 10

    def __init__(self, activation_function: type[nn.Module] = nn.ELU):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            activation_function(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.MaxPool2d(2),
            activation_function(),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.MaxPool2d(3),
            activation_function(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            activation_function(),
            nn.Linear(128, self.N_CLASSES),
        ).to(DEVICE, dtype=FLOAT_DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class EuroSatModel(ParameterizedModule, ClassificationModel):
    N_CLASSES = 10

    def __init__(self, activation_function: type[nn.Module] = nn.ELU):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.MaxPool2d(2),
            activation_function(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.MaxPool2d(2),
            activation_function(),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.MaxPool2d(3),
            activation_function(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            activation_function(),
            nn.Linear(128, self.N_CLASSES),
        ).to(DEVICE, dtype=FLOAT_DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class SVHNModel(ParameterizedModule, ClassificationModel):
    N_CLASSES = 10

    def __init__(self, activation_function: type[nn.Module] = nn.ELU):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            activation_function(),
            nn.Conv2d(16, 32, 3, groups=16),
            nn.MaxPool2d(2),
            activation_function(),
            nn.Conv2d(32, 32, 3, groups=32),
            nn.MaxPool2d(3),
            activation_function(),
            nn.Flatten(),
            nn.Linear(512, 64),
            activation_function(),
            nn.Linear(64, self.N_CLASSES),
        ).to(DEVICE, dtype=FLOAT_DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)
