import numpy as np


def get_loss_sum_up_to_epoch(result: dict[str, np.array], epoch: int) -> float:
    # Proportional to area under the loss curve
    array = result["losses/CCE/training"]
    if len(array) < epoch:
        return np.inf
    else:
        return array[:epoch].sum()
