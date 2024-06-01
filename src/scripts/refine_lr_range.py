"""
Selects the right LR range for each aggregator
Usage:
  refine_lr_range <name>

Arguments:
  <name>             The name of the study in which the experiments belong.

Options:
  -h --help          Show this screen.
"""
import pickle

import numpy as np
import pandas as pd
from docopt import docopt

from jde.settings import LOGS_PATH, RUNS_PATH
from jde.utils import get_loss_sum_up_to_epoch


def main() -> None:
    arguments = docopt(__doc__)
    study_name = arguments["<name>"]
    dir_ = LOGS_PATH / "results" / study_name
    meta_df = pd.read_csv(dir_ / "meta.csv")

    with open(dir_ / "results.pkl", "rb") as f:
        results = pickle.load(f)

    values = []
    for i in range(len(meta_df)):
        name = meta_df.iloc[i]["name"]
        if meta_df.iloc[i]["has_failed"]:
            value = np.inf
        else:
            required_epochs = meta_df.iloc[i]["total_epochs"]
            value = get_loss_sum_up_to_epoch(results[name], required_epochs)
        values.append(value)

    meta_df["value"] = values

    best_lr_exponents = []
    second_best_lr_exponents = []
    min_lr_exponents = []
    max_lr_exponents = []
    group_names = []

    for group_name, df in meta_df.groupby("group_name", sort=False):
        best_idx, second_best_idx = df["value"].sort_values(ascending=True).head(2).index

        best = np.log10(meta_df.loc[best_idx]["optimizer_lr"])
        second_best = np.log10(meta_df.loc[second_best_idx]["optimizer_lr"])
        min_ = min(best, second_best) - 1 / 3
        max_ = max(best, second_best) + 1 / 3

        if max_ - min_ > 1.0001:
            print(f"For group {group_name}, the range is larger than 1.")

        best_lr_exponents.append(best)
        second_best_lr_exponents.append(second_best)
        min_lr_exponents.append(min_)
        max_lr_exponents.append(max_)
        group_names.append(group_name)

    lr_df = pd.DataFrame(
        data={
            "group_name": group_names,
            "best_lr_exp": best_lr_exponents,
            "second_best_lr_exp": second_best_lr_exponents,
            "min_lr_exp": min_lr_exponents,
            "max_lr_exp": max_lr_exponents,
        }
    )

    print(lr_df)

    lr_range = lr_df[["group_name", "min_lr_exp", "max_lr_exp"]]
    path = RUNS_PATH / study_name / "refined_lr_ranges.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lr_range.to_csv(path, index=False)


if __name__ == "__main__":
    main()
