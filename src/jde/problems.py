from __future__ import annotations

import ssl
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.datasets import CIFAR10, KMNIST, MNIST, SVHN, EuroSAT, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from jde.settings import DATA_PATH


@dataclass
class SizeSpec:
    epochs: int
    train_samples: int | None
    test_samples: int | None
    train_split: int = 0
    test_split: int = 0


class Problem:
    """
    Class for defining deep learning optimization problems, with methods for instantiating the
    objects used for training.
    """

    MAX_BS = 128
    USE_DETERMINISTIC_ALGORITHMS: bool = True
    BASE_SIZE_SPECS: dict

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, size_spec: SizeSpec):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = size_spec.epochs
        self.train_samples = size_spec.train_samples
        self.test_samples = size_spec.test_samples
        self.train_split = size_spec.train_split
        self.test_split = size_spec.test_split

        if len(self.train_dataset) < (self.train_split + 1) * self.train_samples:
            raise ValueError(
                f"Not enough data in train dataset to have a split {self.train_split} with "
                f"{self.train_samples} samples."
            )

        if len(self.test_dataset) < (self.test_split + 1) * self.test_samples:
            raise ValueError(
                f"Not enough data in test dataset to have a split {self.test_split} with "
                f"{self.test_samples} samples."
            )

        if self.train_samples is None:
            self._train_sampler = None
            self._train_shuffle = True
        else:
            start = self.train_split * self.train_samples
            stop = (self.train_split + 1) * self.train_samples
            train_indices = range(start, stop)
            self._train_sampler = SubsetRandomSampler(indices=train_indices)
            self._train_shuffle = False

        if self.test_samples is None:
            self._test_sampler = None
            self._test_shuffle = True
        else:
            start = self.test_split * self.test_samples
            stop = (self.test_split + 1) * self.test_samples
            test_indices = range(start, stop)
            self._test_sampler = SubsetRandomSampler(indices=test_indices)
            self._test_shuffle = False

    def __str__(self) -> str:
        return self.__class__.__name__

    def make_train_dataloader(self, train_batch_size: int, drop_last: bool) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            sampler=self._train_sampler,
            drop_last=drop_last,
            shuffle=self._train_shuffle,
        )

    def make_train_dataloader_evaluation(
        self, evaluation_batch_size: int, drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            evaluation_batch_size,
            sampler=self._train_sampler,
            drop_last=drop_last,
            shuffle=self._train_shuffle,
        )

    def make_test_dataloader(self, evaluation_batch_size: int, drop_last: bool) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            evaluation_batch_size,
            sampler=self._test_sampler,
            drop_last=drop_last,
            shuffle=self._test_shuffle,
        )

    @property
    def config(self) -> dict[str:Any]:
        return {
            "problem": str(self),
            "train_dataset": self.train_dataset.__class__.__name__,
            "test_dataset": self.test_dataset.__class__.__name__,
            "total_epochs": self.epochs,
            "n_samples_train": self.train_samples,
            "n_samples_test": self.test_samples,
        }

    def _get_size_spec(self, size_key: str):
        size_spec = self.BASE_SIZE_SPECS[size_key]
        size_spec.train_samples = (size_spec.train_samples // self.MAX_BS) * self.MAX_BS
        size_spec.test_samples = (size_spec.test_samples // self.MAX_BS) * self.MAX_BS
        return size_spec


class Cifar10Problem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=50, train_samples=50000, test_samples=10000),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=5120),
        "faster": SizeSpec(epochs=20, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(10000 // 1024):
        ss = SizeSpec(epochs=20, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        size_spec = self._get_size_spec(size_key)

        transform = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        self.size_key = size_key
        path = DATA_PATH / "CIFAR10"
        super().__init__(
            train_dataset=CIFAR10(root=path, transform=transform, train=True, download=True),
            test_dataset=CIFAR10(root=path, transform=transform, train=False, download=True),
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


class MnistProblem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=10, train_samples=60000, test_samples=10000),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=10000),
        "faster": SizeSpec(epochs=8, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(10000 // 1024):
        ss = SizeSpec(epochs=8, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        size_spec = self._get_size_spec(size_key)

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.size_key = size_key
        super().__init__(
            train_dataset=MNIST(root=DATA_PATH, transform=transform, train=True, download=True),
            test_dataset=MNIST(root=DATA_PATH, transform=transform, train=False, download=True),
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


class KMnistProblem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=10, train_samples=60000, test_samples=10000),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=10000),
        "faster": SizeSpec(epochs=10, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(10000 // 1024):
        ss = SizeSpec(epochs=10, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        size_spec = self._get_size_spec(size_key)
        transform = Compose([ToTensor(), Normalize((0.1918,), (0.3483,))])
        self.size_key = size_key
        super().__init__(
            train_dataset=KMNIST(
                root=DATA_PATH,
                transform=transform,
                train=True,
                download=True,
            ),
            test_dataset=KMNIST(
                root=DATA_PATH,
                transform=transform,
                train=False,
                download=True,
            ),
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


class EuroSatProblem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=10, train_samples=13500, test_samples=13500),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=10000),
        "faster": SizeSpec(epochs=30, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(13500 // 1024):
        ss = SizeSpec(epochs=30, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        ssl._create_default_https_context = ssl._create_unverified_context
        transform = Compose(
            [ToTensor(), Normalize((0.3462, 0.3815, 0.4090), (0.2029, 0.1374, 0.1162))]
        )
        dataset = EuroSAT(
            root=DATA_PATH,
            transform=transform,
            download=True,
        )
        # The train/test split should always be the same
        generator = torch.Generator().manual_seed(0)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [13500, 13500], generator
        )

        size_spec = self._get_size_spec(size_key)

        self.size_key = size_key
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


class FashionMnistProblem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=10, train_samples=60000, test_samples=10000),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=10000),
        "faster": SizeSpec(epochs=25, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(10000 // 1024):
        ss = SizeSpec(epochs=25, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        size_spec = self._get_size_spec(size_key)

        transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
        self.size_key = size_key
        super().__init__(
            train_dataset=FashionMNIST(
                root=DATA_PATH, transform=transform, train=True, download=True
            ),
            test_dataset=FashionMNIST(
                root=DATA_PATH, transform=transform, train=False, download=True
            ),
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


class SVHNProblem(Problem):
    BASE_SIZE_SPECS = {
        "full": SizeSpec(epochs=10, train_samples=73257, test_samples=26032),
        "fast": SizeSpec(epochs=10, train_samples=10240, test_samples=10000),
        "faster": SizeSpec(epochs=25, train_samples=1024, test_samples=1024),
        "tiny": SizeSpec(epochs=1, train_samples=128, test_samples=128),
    }

    for i in range(26032 // 1024):
        ss = SizeSpec(epochs=25, train_samples=1024, test_samples=1024, train_split=i, test_split=i)
        BASE_SIZE_SPECS[f"faster_{i}"] = ss

    def __init__(self, size_key: str):
        self.size_key = size_key
        size_spec = self._get_size_spec(size_key)

        transform = Compose(
            [ToTensor(), Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]
        )
        super().__init__(
            train_dataset=SVHN(root=DATA_PATH, transform=transform, split="train", download=True),
            test_dataset=SVHN(root=DATA_PATH, transform=transform, split="test", download=True),
            size_spec=size_spec,
        )

    @property
    def config(self) -> dict[str:Any]:
        return super().config | {"size_key": self.size_key}


KEY_TO_PROBLEM = {
    "cifar10": Cifar10Problem,
    "mnist": MnistProblem,
    "kmnist": KMnistProblem,
    "fashion_mnist": FashionMnistProblem,
    "euro_sat": EuroSatProblem,
    "svhn": SVHNProblem,
}
