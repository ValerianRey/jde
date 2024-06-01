import numpy as np
from secprint import SectionPrinter as Spt
from torch import Tensor
from torchmetrics import MetricCollection

from jde.metrics import Metrics, MultiBatchWrapper


class Columns:
    SMALL = 12
    MEDIUM = 16
    LARGE = 22


def make_header_base() -> str:
    return (
        "Epoch".ljust(Columns.SMALL)
        + "│ Mode".ljust(Columns.SMALL)
        + "│ Dataset".ljust(Columns.MEDIUM)
        + "│ Batch".ljust(Columns.MEDIUM)
        + "│ LR".ljust(Columns.SMALL)
    )


def make_header_output_metrics(output_metrics: MetricCollection) -> str:
    return "".join([f"│ {metric_name}".ljust(Columns.SMALL) for metric_name in output_metrics])


def make_header_loss_metrics(multiloss_metric: MetricCollection) -> str:
    return "".join(
        [f"│ {loss_function_name}".ljust(Columns.SMALL) for loss_function_name in multiloss_metric]
    )


def make_header_vector_matrix_metrics(vector_matrix_metrics: MultiBatchWrapper):
    return "".join([f"│ {key}".ljust(Columns.SMALL) for key in vector_matrix_metrics.metric])


def print_header(metrics: Metrics) -> None:
    string = make_header_base()
    string += make_header_output_metrics(metrics.output_metrics.metric)
    string += make_header_loss_metrics(metrics.loss_metrics.metric)

    if metrics.vector_matrix_metrics is not None:
        string += make_header_vector_matrix_metrics(metrics.vector_matrix_metrics)

    Spt.print(string, bold=True)


def make_line_base(
    epoch: int,
    epochs: int,
    is_train: bool,
    dataset_name: str,
    batch: int,
    batches: int,
    lr: float | None,
) -> str:
    mode_str = "Train" if is_train else "Evaluate"
    lr_str = "" if lr is None else f"{lr:.3e}"
    return (
        f"[{epoch + 1}/{epochs}]".ljust(Columns.SMALL)
        + f"│ {mode_str}".ljust(Columns.SMALL)
        + f"│ {dataset_name}".ljust(Columns.MEDIUM)
        + f"│ [{batch + 1}/{batches}]".ljust(Columns.MEDIUM)
        + f"│ {lr_str}".ljust(Columns.SMALL)
    )


def make_line_output_metrics(output_metrics: dict[str, Tensor]) -> str:
    return "".join([f"│ {value:.4f}".ljust(Columns.SMALL) for value in output_metrics.values()])


def make_line_loss_metrics(loss_metrics: dict[str, Tensor]) -> str:
    return "".join([f"│ {value:.4f}".ljust(Columns.SMALL) for value in loss_metrics.values()])


def make_line_vector_matrix_metrics(vector_matrix_metrics: list[dict[str, Tensor]]) -> str:
    if len(vector_matrix_metrics) <= 0:
        return ""
    else:
        keys = vector_matrix_metrics[0].keys()

    mean_metrics = {}
    for key in keys:
        all_values_for_key = [batch_metrics[key].item() for batch_metrics in vector_matrix_metrics]
        mean_metrics[key] = np.array(all_values_for_key).mean()

    return "".join([f"│ {value:.4f}".ljust(Columns.SMALL) for value in mean_metrics.values()])


def print_line(
    epoch: int,
    epochs: int,
    is_train: bool,
    dataset_name: str,
    batch: int,
    batches: int,
    lr: float | None,
    metrics: Metrics,
    persistent: bool = False,
) -> None:
    string = make_line_base(epoch, epochs, is_train, dataset_name, batch, batches, lr)
    output_metrics = metrics.output_metrics.compute()
    string += make_line_output_metrics(output_metrics)
    string += make_line_loss_metrics(metrics.loss_metrics.compute())

    if metrics.vector_matrix_metrics is not None and is_train:
        string += make_line_vector_matrix_metrics(metrics.vector_matrix_metrics.compute())

    Spt.print(string, rewrite=True, end="\n" if persistent else "")


def dict_to_str(dict_: dict) -> str:
    """
    This returns a string of the form {repr(key): str(value), ...}, instead of the usual
    {repr(key), repr(value), ...} that dict.__str__ gives.
    """

    elements = []
    for key, value in dict_.items():
        if key == "lr":
            elements.append(f"{repr(key)}: 10^{np.log10(value):.2f}")
        else:
            elements.append(f"{repr(key)}: {str(value)}")
    return "{" + ", ".join(elements) + "}"
