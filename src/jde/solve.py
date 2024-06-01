import os
from time import time
from traceback import print_exception

import torch
import wandb
from secprint import SectionPrinter as Spt
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchjd.module_wrapper import ModuleWrapper

from jde.metrics import Metrics
from jde.printing import print_header, print_line
from jde.problems import Problem
from jde.settings import DEVICE, FLOAT_DTYPE, LOGS_PATH
from jde.solutions import Solution


def solve(
    problem: Problem,
    solution: Solution,
    study_name: str,
    wandb_mode: str,
    seed: int = 0,
    print_train_time: bool = False,
) -> bool:
    # Fix seed to make some experiments in identical setups
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(problem.USE_DETERMINISTIC_ALGORITHMS)

    solution_name = str(solution)
    model_wrapper = solution.make_wrapper()
    optimizer = solution.make_optimizer(model_wrapper)
    lr_scheduler = solution.make_lr_scheduler(optimizer)
    train_bs = solution.train_batch_size
    eval_bs = solution.evaluation_batch_size
    drop_last_batch = solution.drop_last_batch

    train_dataloader = problem.make_train_dataloader(train_bs, drop_last_batch)
    train_dataloader_evaluation = problem.make_train_dataloader_evaluation(eval_bs, drop_last_batch)
    test_dataloader = problem.make_test_dataloader(eval_bs, drop_last_batch)

    os.environ["WANDB_SILENT"] = "true"

    config = problem.config | solution.config | {"seed": seed}
    name = f"{str(solution)} - seed: {seed}"
    if "size_key" in problem.config:
        size_key = problem.config["size_key"]
        name += f" - size_key: {size_key}"

    wandb.init(
        dir=LOGS_PATH,
        project="jde2",
        config=config,
        name=name,
        group=study_name,
        mode=wandb_mode,
    )

    is_success = True

    title = " - ".join([str(problem), solution_name])
    with Spt(title, color=solution.header_color):
        metrics = solution.metrics
        print_header(metrics)

        for epoch in range(problem.epochs):
            try:
                printing_dict = {
                    "epoch": epoch,
                    "epochs": problem.epochs,
                    "dataset_name": f"Train ({problem.train_samples})",
                }
                metrics.mode = "training"
                start_time = time()
                _train(
                    model_wrapper,
                    train_dataloader,
                    optimizer=optimizer,
                    printing_dict=printing_dict,
                    metrics=metrics,
                )
                if print_train_time:
                    Spt.print(f"Elapsed time: {time() - start_time:.2f} seconds.")

                metrics.log_and_commit_per_batch_metrics(epoch=epoch, training=True)
                metrics.log_per_epoch_metrics(epoch=epoch, training=True, commit=False)

                metrics.mode = "train_eval"
                _evaluate(
                    model_wrapper,
                    train_dataloader_evaluation,
                    printing_dict=printing_dict,
                    metrics=metrics,
                )
                metrics.log_per_epoch_metrics(epoch=epoch, training=False, commit=False)

                printing_dict["dataset_name"] = f"Test ({problem.test_samples})"
                metrics.mode = "test_eval"
                _evaluate(
                    model_wrapper,
                    test_dataloader,
                    printing_dict=printing_dict,
                    metrics=metrics,
                )
                metrics.log_per_epoch_metrics(epoch=epoch, training=False, commit=True)
                lr_scheduler.step()
            except Exception as e:
                print_exception(e)
                is_success = False
                break
            finally:
                metrics.reset_all()

    wandb.finish()
    return is_success


def _train(
    module_wrapper: ModuleWrapper,
    dataloader: DataLoader,
    optimizer: Optimizer,
    printing_dict: dict,
    metrics: Metrics,
) -> None:
    module_wrapper.train()
    lr = optimizer.param_groups[0]["lr"]

    for i, (x, target) in enumerate(dataloader):
        x, target = _move_data(x, target)
        loss_hierarchy = module_wrapper(x, target)

        optimizer.zero_grad()
        loss_hierarchy.backward()
        optimizer.step()

        print_line(
            epoch=printing_dict["epoch"],
            epochs=printing_dict["epochs"],
            is_train=True,
            dataset_name=printing_dict["dataset_name"],
            batch=i,
            batches=len(dataloader),
            lr=lr,
            metrics=metrics,
            persistent=i == len(dataloader) - 1,
        )


@torch.no_grad()
def _evaluate(
    module_wrapper: ModuleWrapper,
    dataloader: DataLoader,
    printing_dict: dict,
    metrics: Metrics,
) -> None:
    module_wrapper.eval()

    for i, (x, target) in enumerate(dataloader):
        x, target = _move_data(x, target)
        # The metrics are computed by the hooks when calling module_wrapper.
        # We thus don't need to store its result
        module_wrapper(x, target)

        print_line(
            epoch=printing_dict["epoch"],
            epochs=printing_dict["epochs"],
            is_train=False,
            dataset_name=printing_dict["dataset_name"],
            batch=i,
            batches=len(dataloader),
            lr=None,
            metrics=metrics,
            persistent=i == len(dataloader) - 1,
        )


def _move_data(x: Tensor, target: Tensor | list[Tensor]) -> tuple[Tensor, Tensor | list[Tensor]]:
    """Cast data to the right dtype and move it to the right device"""

    x = x.to(DEVICE, dtype=FLOAT_DTYPE)

    if isinstance(target, list):
        for i in range(len(target)):
            target[i] = _move_target_tensor(target[i])
    else:
        target = _move_target_tensor(target)

    return x, target


def _move_target_tensor(target: Tensor) -> Tensor:
    target = target.to(DEVICE)

    if target.dtype in {torch.float16, torch.float32, torch.float64}:
        target = target.to(dtype=FLOAT_DTYPE)

    return target
