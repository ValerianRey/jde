"""
Profiles memory and computation time of one training epoch for each method.
Usage:
  profiling_study [--dataset=<key>] [--batch-size=<n>] [--size=<key>]

Options:
  -h --help             Show this screen.
  --dataset=<key>       Dataset to profile [default: cifar10].
  --batch-size=<n>      Batch size [default: 32].
  --size=<key>          Problem size key (tiny/faster/fast/full) [default: faster].
"""
import gc

import torch
from docopt import docopt
from torch import nn
from torch.optim import SGD
from torch.profiler import ProfilerActivity, profile
from torchjd.aggregation import UPGrad, UPGradWeighting

from jde.architectures import Cifar10Model, EuroSatModel, MnistModel, SVHNModel
from jde.criteria import LossListCriterion, SplitTensorCriterion
from jde.loss_combiners import SqueezeCombiner
from jde.loss_functions import NamedModule
from jde.module_wrappers import FullJDWrapper, GramianJDWrapper, LossCombinationGDWrapper
from jde.problems import KEY_TO_PROBLEM
from jde.settings import DEVICE, FLOAT_DTYPE, LOGS_PATH

DATASET_KEY_TO_ARCHITECTURE = {
    "svhn": SVHNModel,
    "cifar10": Cifar10Model,
    "euro_sat": EuroSatModel,
    "mnist": MnistModel,
}

TRACES_DIR = LOGS_PATH / "profiling"


def profile_epoch(
    method_name: str,
    module_wrapper: nn.Module,
    optimizer,
    dataloader,
    dataset_key: str,
    batch_size: int,
) -> None:
    """
    Profiles memory and computation time of one training epoch.

    :param method_name: Name of the method being profiled (used for the output path).
    :param module_wrapper: The model wrapper to profile.
    :param optimizer: The optimizer to use.
    :param dataloader: DataLoader for the training data.
    :param dataset_key: Dataset identifier (used for the output filename).
    :param batch_size: Batch size (used for the output filename).
    """
    print(f"Profiling {method_name} on {dataset_key} (BS={batch_size}) on {DEVICE}:")

    _run_epoch(module_wrapper, optimizer, dataloader)
    module_wrapper.zero_grad()
    _clear_memory()

    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
        with_stack=True,
    ) as prof:
        _run_epoch(module_wrapper, optimizer, dataloader)

    output_dir = TRACES_DIR / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{dataset_key}-bs{batch_size}-{DEVICE.type}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"  Trace saved to {trace_path}")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))


def _run_epoch(module_wrapper: nn.Module, optimizer, dataloader) -> None:
    module_wrapper.train()
    for x, target in dataloader:
        x = x.to(DEVICE, dtype=FLOAT_DTYPE)
        target = target.to(DEVICE)
        loss_hierarchy = module_wrapper(x, target)
        optimizer.zero_grad()
        loss_hierarchy.backward()
        optimizer.step()


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_autograd_wrapper(architecture: type) -> LossCombinationGDWrapper:
    cce = NamedModule(nn.CrossEntropyLoss(), name="CCE")
    criterion = LossListCriterion(loss_functions=[cce])
    model = architecture()
    return LossCombinationGDWrapper(
        model=model, criterion=criterion, loss_combiner=SqueezeCombiner()
    )


def _make_autojac_wrapper(architecture: type) -> FullJDWrapper:
    cce = NamedModule(nn.CrossEntropyLoss(), name="CCE")
    criterion = SplitTensorCriterion(loss_function=cce, dim=0)
    model = architecture()
    return FullJDWrapper(model=model, criterion=criterion, aggregator=UPGrad())


def _make_autogram_wrapper(architecture: type) -> GramianJDWrapper:
    cce = NamedModule(nn.CrossEntropyLoss(), name="CCE")
    criterion = SplitTensorCriterion(loss_function=cce, dim=0)
    model = architecture()
    return GramianJDWrapper(
        model=model, criterion=criterion, gramian_weighting=UPGradWeighting(), batch_dim=0
    )


METHODS = {
    "autograd": _make_autograd_wrapper,
    "autojac": _make_autojac_wrapper,
    "autogram": _make_autogram_wrapper,
}


def main() -> None:
    args = docopt(__doc__)
    dataset_key = args["--dataset"] or "cifar10"
    batch_size = int(args["--batch-size"] or 32)
    size_key = args["--size"] or "faster"

    problem = KEY_TO_PROBLEM[dataset_key](size_key)
    architecture = DATASET_KEY_TO_ARCHITECTURE[dataset_key]
    dataloader = problem.make_train_dataloader(batch_size, drop_last=True)

    for method_name, make_wrapper in METHODS.items():
        module_wrapper = make_wrapper(architecture)
        optimizer = SGD(module_wrapper.parameters(), lr=1e-3)
        profile_epoch(method_name, module_wrapper, optimizer, dataloader, dataset_key, batch_size)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
