# The code of this file was partly adapted from
# https://github.com/SamsungLabs/MTL/tree/master/code/optim/aligned.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2022 Samsung
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import Tensor
from torch.linalg import LinAlgError

from torchjd.aggregation.bases import Weighting


class AlignedMTLWrapper(Weighting):
    r"""
    Wrapper of :class:`~torchjd.aggregation.bases.Weighting` that corrects the extracted
    weights with the balance transformation defined in Algorithm 1 of `Independent Component
    Alignment for Multi-Task Learning
    <https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf>`_.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases.Weighting`
        responsible for extracting weight vectors from the input matrices.

    .. admonition::
        Example

        Use AlignedMTL to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import (
        ...     WeightedAggregator,
        ...     MeanWeighting,
        ...     AlignedMTLWrapper,
        ... )
        >>>
        >>> W = AlignedMTLWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.2133, 0.9673, 0.9673])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([0.5591, 0.4083])

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/SamsungLabs/MTL/tree/master/code/optim/aligned>`_.
    """

    def __init__(self, weighting: Weighting):
        super().__init__()
        self.weighting = weighting

    def forward(self, matrix: Tensor) -> Tensor:
        w = self.weighting(matrix)

        G = matrix.T
        B = self._compute_balance_transformation(G)
        alpha = B @ w

        return alpha

    @staticmethod
    def _compute_balance_transformation(G: Tensor) -> Tensor:
        M = G.T @ G

        try:
            lambda_, V = torch.linalg.eigh(M, UPLO="U")  # More modern equivalent to torch.symeig
        except LinAlgError:  # This can happen when the matrix has extremely large values
            identity = torch.eye(len(M), dtype=M.dtype, device=M.device)
            return identity

        tol = torch.max(lambda_) * len(M) * torch.finfo().eps
        rank = sum(lambda_ > tol)

        if rank == 0:
            identity = torch.eye(len(M), dtype=M.dtype, device=M.device)
            return identity

        order = torch.argsort(lambda_, dim=-1, descending=True)
        lambda_, V = lambda_[order][:rank], V[:, order][:, :rank]

        sigma_inv = torch.diag(1 / lambda_.sqrt())
        lambda_R = lambda_[-1]
        B = lambda_R.sqrt() * V @ sigma_inv @ V.T
        return B

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weighting={repr(self.weighting)})"

    def __str__(self) -> str:
        return f"AlignedMTL {self.weighting}"
