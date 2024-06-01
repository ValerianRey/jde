from torchjd.module_wrapper.base import ModuleWrapper
from torchjd.tensor import TensorHierarchy
from torchjd.transform import Scaling


class GradientScalingLayer(ModuleWrapper):
    """
    .. hint::
        This can be used to implement gradient reversal, as introduced in `Unsupervised Domain
        Adaptation by Backpropagation <https://arxiv.org/pdf/1409.7495.pdf>`_.
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, input: TensorHierarchy) -> TensorHierarchy:
        scaling = Scaling({tensor: self.scale for tensor in input.data})
        return TensorHierarchy(input.data, input.transform << scaling)
