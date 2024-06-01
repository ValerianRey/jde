"""
Downloads all runs of a given study from wandb.
Usage:
  download_study <name>

Arguments:
  <name>             The name of the study in which the experiments belong.

Options:
  -h --help          Show this screen.
"""
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from docopt import docopt
from wandb import Api
from wandb.apis.public import Run

from jde.settings import LOGS_PATH

PER_EPOCH_KEYS = [
    "losses/CCE/test_eval",
    "losses/CCE/train_eval",
    "losses/CCE/training",
    "output/Acc/test_eval",
    "output/Acc/train_eval",
    "output/Acc/training",
    "weight/Average",
]

PER_BATCH_KEYS = [
    "losses/CCE/training (per-batch)",
    "gradient-jacobian/Cosine similarity to mean (per-batch)",
]


def main():
    arguments = docopt(__doc__)
    study_name = arguments["<name>"]

    save_path = LOGS_PATH / "results" / study_name

    api = Api()
    entity, project = "multeam-rocket", "jde2"
    runs = api.runs(entity + "/" + project, filters={"group": study_name})

    if len(runs) == 0:
        raise ValueError(f"No run found for study '{study_name}'.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    configs, names, group_names, have_failed = [], [], [], []
    name_to_results = {}

    for i, run in enumerate(runs):
        print(f"[{i+1}/{len(runs)}]")

        name, config, summary = run.name, get_config(run), get_summary(run)
        group_name = config["aggregator"]
        has_failed = len(summary) == 0

        configs.append(config)
        names.append(name)
        have_failed.append(has_failed)
        group_names.append(group_name)

        if has_failed:
            continue

        sh = run.scan_history()
        rows = []
        for row in sh:
            rows.append(row)

        df = pd.DataFrame(rows)

        per_epoch_results = {k: process_series(df[k]) for k in PER_EPOCH_KEYS if k in df.columns}
        per_batch_results = {k: process_series(df[k]) for k in PER_BATCH_KEYS if k in df.columns}
        results = per_batch_results | per_epoch_results
        name_to_results[name] = results

    config_metadata = list_of_dicts_to_dict_of_lists(configs)
    meta_df = pd.DataFrame(
        {"name": names, "group_name": group_names, "has_failed": have_failed} | config_metadata
    )

    # Reverse the order of the rows to have the first experiments appear first
    meta_df = meta_df[::-1]

    meta_df.to_csv(save_path / "meta.csv", index=False)
    with open(save_path / "results.pkl", "wb") as f:
        pickle.dump(name_to_results, f)


def process_series(series: pd.Series) -> np.ndarray:
    """
    At first, the series can contain actual values, NaN values (which are missing values, for
    instance because the metric was not reported during per-batches, so it misses from all rows
    except per-epoch rows), and 'NaN' values (which are representing an actual nan for instance when
    the loss diverged). This starts by removing missing values, then casts everything to numpy type,
    then casts to float64, which turns the NaN strings into np.nan.
    """
    return series.dropna().to_numpy().astype(np.float64)


def list_of_dicts_to_dict_of_lists(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    all_keys = sorted(list(set().union(*[set(d.keys()) for d in list_of_dicts])))
    dict_of_lists = {key: [] for key in all_keys}

    for d in list_of_dicts:
        for key in all_keys:
            value = d.get(key, "None")
            dict_of_lists[key].append(value)

    return dict_of_lists


def get_config(run: Run) -> dict:
    return {k: v for k, v in run.config.items() if not k.startswith("_")}


def get_summary(run: Run) -> dict:
    return {k: v for k, v in run.summary.items() if not k.startswith("_")}


if __name__ == "__main__":
    main()
