"""
Runs an IWRM study on a given dataset.
Usage:
  iwrm_study <dataset> <size> <n_lr> [--lr-csv-path=<str>] [--batch-size=<int>] [--epochs=<int>] [--optimizer=<str>] [--seed=<int>] [--wandb-mode=<str>] [--name=<str>]

Arguments:
  <dataset>             The string defining the dataset (cifar10, mnist, ...)
  <size>                The string defining the size of the problem (tiny, faster, fast, full, ...)
  <n_lr>                Number of learning rates to try from the range.
Options:
  -h --help             Show this screen.
  --lr-csv-path=<str>   Path leading to the file giving the initial range of learning rates to use.
                        Relative to the runs/ folder. [default: default_lr_ranges.csv].
  --batch-size=<int>    Batch size to use. [default: 32].
  --epochs=<int>        Use this parameter to override the default number of epochs. [default: -1]
  --optimizer=<str>     Which optimizer to use ("SGD" or "Adam"). [default: SGD].
  --name=<str>          The name of the study in which the experiments belong.
                        Only used for logging [default: ]
  --wandb-mode=<str>    The mode to use for Weights and Biases logging. [default: disabled]
  --seed=<int>          Value of the seed to use for random number generation. This can be changed
                        to compute the standard deviation of the results over the randomness of the
                        whole process. [default: 0]
"""
import numpy as np
import pandas as pd
from docopt import docopt
from secprint import SectionPrinter as Spt
from torch.optim import SGD, Adam

from jde.architectures import Cifar10Model, EuroSatModel, MnistModel, SVHNModel
from jde.problems import KEY_TO_PROBLEM
from jde.settings import RUNS_PATH
from jde.solution_generators import generate_iwrm_solutions
from jde.solve import solve

DATASET_KEY_TO_ARCHITECTURE = {
    "mnist": MnistModel,
    "fashion_mnist": MnistModel,
    "kmnist": MnistModel,
    "cifar10": Cifar10Model,
    "euro_sat": EuroSatModel,
    "svhn": SVHNModel,
}


def main() -> None:
    Spt.set_automatic_skip(True)
    Spt.set_default_header("┃ ")

    arguments = docopt(__doc__)

    dataset_key = arguments["<dataset>"]
    size_key = arguments["<size>"]
    n_lr = int(arguments["<n_lr>"])
    lr_csv_path = arguments["--lr-csv-path"]
    batch_size = int(arguments["--batch-size"])
    optimizer_str = arguments["--optimizer"]
    name = arguments["--name"]
    wandb_mode = arguments["--wandb-mode"]
    seed = int(arguments["--seed"])
    epochs = int(arguments["--epochs"])

    problem = KEY_TO_PROBLEM[dataset_key](size_key)
    if epochs != -1:
        problem.epochs = epochs

    architecture = DATASET_KEY_TO_ARCHITECTURE[dataset_key]

    if optimizer_str == "SGD":
        optimizer_class = SGD
        optimizer_kwargs = {"lr": np.nan}
    elif optimizer_str == "Adam":
        optimizer_class = Adam
        optimizer_kwargs = {"lr": np.nan, "betas": (0.9, 0.999), "eps": 1e-8}
    else:
        raise ValueError(f"Wrong optimizer_str {optimizer_str}.")

    path = RUNS_PATH / lr_csv_path
    lr_df = pd.read_csv(path)

    solutions = generate_iwrm_solutions(
        architecture=architecture,
        lr_df=lr_df,
        n_lr=n_lr,
        batch_size=batch_size,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
    )

    n_experiments = 0
    n_successes = 0
    for solution in solutions:
        n_experiments += 1
        n_successes += solve(
            problem=problem,
            solution=solution,
            study_name=name,
            wandb_mode=wandb_mode,
            seed=seed,
        )

    color = "green" if n_successes == n_experiments else "red"
    Spt.print(f"\n{n_successes}/{n_experiments} successful experiments", color=color, bold=True)


if __name__ == "__main__":
    main()
