"""
Extracts the run with the best loss AUC for each aggregator and each subset.
Usage:
  extract_best_runs <name-prefix>

Arguments:
  <name-prefix>      The prefix of name of the study in which the experiments belong (not including
                     the seed/subset number)
Options:
  -h --help          Show this screen.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from docopt import docopt

from jde.settings import LOGS_PATH
from jde.utils import get_loss_sum_up_to_epoch


def main() -> None:
    arguments = docopt(__doc__)
    study_prefix = arguments["<name-prefix>"]
    res = LOGS_PATH / "results"

    # Consider only folders that start with the given prefix and are followed by numbers and spaces
    dirs = [
        item
        for item in res.iterdir()
        if item.is_dir()
        and item.name.startswith(study_prefix)
        and item.name[len(study_prefix) :].replace(" ", "").isdigit()
    ]

    results = {}
    meta_df_list = []

    for dir_ in dirs:
        sub_results, sub_meta_df = select_from_substudy(dir_)
        results.update(sub_results)
        meta_df_list.append(sub_meta_df)

    meta_df = pd.concat(meta_df_list).reset_index(drop=True)

    save_dir = LOGS_PATH / "results" / f"{study_prefix} combined"
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(save_dir / "meta.csv", index=False)
    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)


def select_from_substudy(dir_: Path) -> tuple[dict, pd.DataFrame]:
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

    sub_results = {}
    sub_meta_df_rows = []

    for group_name, df in meta_df.groupby("group_name", sort=False):
        (best_idx,) = df["value"].sort_values(ascending=True).head(1).index
        name = meta_df.loc[best_idx]["name"]
        sub_results[name] = results[name]
        sub_meta_df_rows.append(meta_df.loc[best_idx])

    sub_meta_df = pd.DataFrame(sub_meta_df_rows).reset_index(drop=True)
    return sub_results, sub_meta_df


if __name__ == "__main__":
    main()
