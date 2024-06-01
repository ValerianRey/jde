"""
Compares the speed of various methods on various datasets.
Usage:
  speed_study

Options:
  -h --help             Show this screen.
"""
import itertools

import numpy as np
import pandas as pd
from docopt import docopt
from secprint import SectionPrinter as Spt
from torch.optim import SGD

from jde.architectures import Cifar10Model, EuroSatModel, MnistModel, SVHNModel
from jde.problems import KEY_TO_PROBLEM
from jde.settings import RUNS_PATH
from jde.solution_generators import generate_gd_solution, generate_iwrm_solutions
from jde.solve import solve

DATASET_KEY_TO_ARCHITECTURE = {
    "svhn": SVHNModel,
    "cifar10": Cifar10Model,
    "euro_sat": EuroSatModel,
    "mnist": MnistModel,
}

PROBLEM_KEYS = ["svhn", "cifar10", "euro_sat", "mnist"]


def main() -> None:
    Spt.set_automatic_skip(True)
    Spt.set_default_header("┃ ")
    Spt.set_max_depth(1)

    _ = docopt(__doc__)

    lr_df = pd.read_csv(RUNS_PATH / "default_lr_ranges.csv")

    for key in PROBLEM_KEYS:
        problem = KEY_TO_PROBLEM[key]("faster")
        problem.epochs = 1

        architecture = DATASET_KEY_TO_ARCHITECTURE[key]
        optimizer_class = SGD

        gd_solution_generator = generate_gd_solution(architecture, 1e-5)

        solutions = generate_iwrm_solutions(
            architecture=architecture,
            lr_df=lr_df,
            n_lr=1,
            batch_size=32,
            optimizer_class=optimizer_class,
            optimizer_kwargs={"lr": np.nan},  # Will be overridden
        )

        n_experiments = 0
        n_successes = 0
        for solution in itertools.chain(gd_solution_generator, solutions):
            n_experiments += 1
            n_successes += solve(
                problem=problem,
                solution=solution,
                study_name="",
                wandb_mode="disabled",
                print_train_time=True,
            )

        color = "green" if n_successes == n_experiments else "red"
        Spt.print(f"\n{n_successes}/{n_experiments} successful experiments", color=color, bold=True)


if __name__ == "__main__":
    main()
