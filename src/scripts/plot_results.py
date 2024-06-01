"""
Plots the results of a study.
Usage:
  plot_results <name> [--epochs=<int>] [--minimal] [--log] [--alpha=<float>] [--adam]

Arguments:
  <name>             The name of the study in which the experiments belong.

Options:
  -h --help          Show this screen.
  --epochs=<int>     The number of epochs to plot. [default: -1]
  --minimal          Whether to only plot UPGrad and Mean rather than everything. [default: False]
  --log              Whether to set the y-axis of the loss plots in log scale. [default: False]
  --alpha=<float>    Value of smoothing to use. 1.0 means no smoothing. [default: 0.025]
  --adam             If Adam was used as the optimizer, this disables the "(SGD)" in the A_mean
                     caption. [default: False]
"""
import pickle

import matplotlib
import numpy as np
import pandas as pd
from docopt import docopt
from matplotlib import pyplot as plt

from jde.settings import LOGS_PATH

ITERATIONS_PER_EPOCH = 16
X_AXIS_MARGIN_PROP = 0.01

LOSS_KEY = "losses/CCE/training (per-batch)"
SIMILARITY_KEY = "gradient-jacobian/Cosine similarity to mean (per-batch)"

NAME_TO_COLOR = {
    "UPGrad Mean": "tab:green",
    "Mean": (0.3, 0.3, 0.3),
    "DualProj Mean": "tab:blue",
    "MGDA": "tab:red",
    "PCGrad": "tab:purple",
    "Random": "dodgerblue",
    "NashMTL": "#a552e6",
    "GradDrop": "#f97306",
    "CAGrad0.5": "#ff6163",
    "IMTLG": "deepskyblue",
    "AlignedMTL Mean": "#e6cb17",
}


def main() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    arguments = docopt(__doc__)
    study_name = arguments["<name>"]
    log_y = arguments["--log"]
    alpha = float(arguments["--alpha"])
    minimal = arguments["--minimal"]
    epochs = int(arguments["--epochs"])

    dir_ = LOGS_PATH / "results" / study_name
    meta_df = pd.read_csv(dir_ / "meta.csv")

    if (
        meta_df["total_epochs"].nunique() != 1
        or meta_df["train_bs"].nunique() != 1
        or meta_df["n_samples_train"].nunique() != 1
    ):
        raise ValueError(
            "Columns total_epochs, train_bs and n_samples_train should have a unique value"
        )

    if epochs == -1:
        epochs = meta_df["total_epochs"].iloc[0]
    iterations_per_epoch = int(meta_df["n_samples_train"].iloc[0] / meta_df["train_bs"].iloc[0])
    iterations = epochs * iterations_per_epoch

    with open(dir_ / "results.pkl", "rb") as f:
        results = pickle.load(f)

    default_groups = [
        "Mean",
        "UPGrad Mean",
    ]

    if minimal:
        plot_id_to_groups = {1: []}
    else:
        plot_id_to_groups = {
            1: [
                "MGDA",
                "AlignedMTL Mean",
                "PCGrad",
                "DualProj Mean",
            ],
            2: [
                "Random",
                "GradDrop",
            ],
            3: [
                "CAGrad0.5",
                "IMTLG",
                "NashMTL",
            ],
        }

    for plot_id in plot_id_to_groups.keys():
        group_names = default_groups + plot_id_to_groups[plot_id]

        group_name_to_loss_array = {}
        group_name_to_similarity_array = {}

        for group_name in group_names:
            df = meta_df[meta_df["group_name"] == group_name]

            names = df["name"].to_list()
            loss_arrays = []
            similarity_arrays = []
            for name in names:
                try:
                    loss_array = results[name][LOSS_KEY][:iterations]
                except KeyError:
                    print(f"A run of {group_name} has failed (no data available at all)")
                    continue
                if len(loss_array) < iterations:
                    print(f"A run of {group_name} has failed")
                    continue

                loss_arrays.append(loss_array)

                if SIMILARITY_KEY in results[name]:
                    similarity_array = results[name][SIMILARITY_KEY][:iterations]
                elif group_name == "Average":
                    similarity_array = np.ones_like(loss_array)
                else:
                    similarity_array = np.empty_like(loss_array)
                    similarity_array[:] = np.nan

                similarity_arrays.append(similarity_array)

            group_name_to_loss_array[group_name] = np.stack(loss_arrays)
            group_name_to_similarity_array[group_name] = np.stack(similarity_arrays)

        plot_losses(group_name_to_loss_array, dir_, plot_id, log_y, alpha)
        plot_similarities(group_name_to_similarity_array, dir_, plot_id, alpha)


def preprocess(data: np.ndarray, alpha: float) -> np.ndarray:
    data = pd.Series(data).ewm(alpha=alpha).mean().to_numpy()
    return data


def plot_losses(
    group_name_to_data: dict[str, np.ndarray], dir_, plot_id: int, log_y: bool, alpha: float
) -> None:
    fig, ax = plt.subplots(1, figsize=(5, 4), dpi=360)
    plot_signals(ax, group_name_to_data, alpha)

    ax.legend(fontsize=10, loc="upper right", ncols=2, columnspacing=0.5)
    ax.set_ylabel("Categorical cross-entropy")
    ax.set_xlabel("Iteration")
    if log_y:
        ax.set_yscale("log")

    ax.grid(color="grey", linestyle=":", linewidth=0.5)

    key = list(group_name_to_data.keys())[0]
    data = group_name_to_data[key]
    n_iterations = data.shape[1]
    x_start = -n_iterations * X_AXIS_MARGIN_PROP
    x_stop = n_iterations * (1 + X_AXIS_MARGIN_PROP)
    ax.set_xlim([x_start, x_stop])

    plt.savefig(dir_ / f"train_losses_{plot_id}.pdf", bbox_inches="tight", dpi=fig.dpi)


def plot_similarities(group_name_to_data: dict[str, np.ndarray], dir_, plot_id: int, alpha) -> None:
    fig, ax = plt.subplots(1, figsize=(5, 4), dpi=360)
    plot_signals(ax, group_name_to_data, alpha)

    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Iteration")
    ax.grid(color="grey", linestyle=":", linewidth=0.5)
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_ylim([0, 1.05])

    key = list(group_name_to_data.keys())[0]
    data = group_name_to_data[key]
    n_iterations = data.shape[1]
    x_start = -n_iterations * X_AXIS_MARGIN_PROP
    x_stop = n_iterations * (1 + X_AXIS_MARGIN_PROP)
    ax.set_xlim([x_start, x_stop])

    plt.savefig(dir_ / f"similarities_{plot_id}.pdf", bbox_inches="tight", dpi=fig.dpi)


def plot_signals(ax, group_name_to_data: dict[str, np.ndarray], alpha: float) -> None:
    # markers = ["none", "*", ">", "<", "v", "^", "p", "h"]
    markers = ["none"] * 8
    markersizes = [5, 7, 5, 5, 5, 5, 5, 5]
    linestyles = ["--"] + ["-"] * 7

    arguments = docopt(__doc__)
    adam = arguments["--adam"]

    name_mapping = {
        "UPGrad Mean": "UPGrad",
        "DualProj Mean": "DualProj",
        "CAGrad0.5": "CAGrad(c=0.5)",
        "IMTLG": "IMTL-G",
        "AlignedMTL Mean": "Aligned-MTL",
        "NashMTL": "Nash-MTL",
        "Random": "RGW",
    }

    for (name, data), marker, markersize, linestyle in zip(
        group_name_to_data.items(), markers, markersizes, linestyles
    ):
        color = NAME_TO_COLOR[name]

        mean_data = data.mean(axis=0)
        mean_std_error = np.std(data, ddof=1, axis=0) / np.sqrt(data.shape[0])

        mean_data = preprocess(mean_data, alpha)
        mean_std_error = preprocess(mean_std_error, alpha)

        if name == "Mean" and not adam:
            suffix = " (SGD)"
        elif name == "UPGrad Mean":
            suffix = " (ours)"
        else:
            suffix = ""

        if name in name_mapping:
            name = name_mapping[name]

        name = r"$\mathcal{A}_{\text{" + str(name) + "}}$" + suffix

        ax.plot(
            mean_data,
            label=name,
            marker=marker,
            markevery=100,
            markersize=markersize,
            markeredgecolor="black",
            markeredgewidth=0.2,
            alpha=0.75,
            linestyle=linestyle,
            color=color,
        )

        ax.fill_between(
            np.arange(len(mean_data)),
            mean_data - mean_std_error,
            mean_data + mean_std_error,
            alpha=0.12,
            color=color,
            linewidth=0.5,
        )


if __name__ == "__main__":
    main()
