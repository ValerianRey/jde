from torch import Tensor


def _move_dim_to_front(t: Tensor, dim: int) -> Tensor:
    """
    Returns a Tensor view in which dimension ``dim`` is moved in front of the other dimensions.
    Example: if ``t`` is of dimension 4 and ``dim`` is ``2``, axes order will be ``[2, 0, 1, 3]``.
    """

    axes_order = [dim] + list(range(dim)) + list(range(dim + 1, t.dim()))
    return t.permute(axes_order)


def _move_dim_back(t: Tensor, dim: int) -> Tensor:
    """
    Returns a Tensor view in which the dimensions are permuted back to what they were before the
    call to ``_move_dim_to_front``.
    """

    axes_order = list(range(t.dim()))[1:]
    axes_order.insert(dim, 0)
    return t.permute(axes_order)
