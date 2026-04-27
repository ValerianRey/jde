# Aggregators
from torchjd.aggregation import (
    IMTLG,
    MGDA,
    AlignedMTL,
    AlignedMTLWeighting,
    CAGrad,
    CAGradWeighting,
    DualProj,
    DualProjWeighting,
    GradDrop,
    IMTLGWeighting,
    Mean,
    MeanWeighting,
    MGDAWeighting,
    NashMTL,
    PCGrad,
    PCGradWeighting,
    Random,
    RandomWeighting,
    Sum,
    UPGrad,
    UPGradWeighting,
)
from torchmetrics import MeanMetric, MetricCollection

from jde.hooks import make_aggregator_hook, make_weighting_hook
from jde.metrics import CosineSimilarityToMatrixMean, MultiBatchWrapper
from jde.settings import DEVICE

upgrad = UPGrad()
mean = Mean()
sum_ = Sum()
mgda = MGDA()
rgw = Random()
dualproj = DualProj()
pcgrad = PCGrad()
imtlg = IMTLG()
graddrop = GradDrop()
alignedm = AlignedMTL()
nashmtl_32 = NashMTL(n_tasks=32, optim_niter=1)
cagrad_0_5 = CAGrad(c=0.5)

# Aggregation Metrics
output_direction_metrics = MetricCollection({})
gradient_jacobian_metrics = MultiBatchWrapper(
    MetricCollection(
        {
            "Cosine similarity to mean": CosineSimilarityToMatrixMean(),
        }
    )
)

_aggregator_hook = make_aggregator_hook(output_direction_metrics, gradient_jacobian_metrics)

upgrad.register_forward_hook(_aggregator_hook)
mean.register_forward_hook(_aggregator_hook)
sum_.register_forward_hook(_aggregator_hook)
mgda.register_forward_hook(_aggregator_hook)
rgw.register_forward_hook(_aggregator_hook)
dualproj.register_forward_hook(_aggregator_hook)
pcgrad.register_forward_hook(_aggregator_hook)
imtlg.register_forward_hook(_aggregator_hook)
graddrop.register_forward_hook(_aggregator_hook)
alignedm.register_forward_hook(_aggregator_hook)
nashmtl_32.register_forward_hook(_aggregator_hook)
cagrad_0_5.register_forward_hook(_aggregator_hook)

# Weight Metrics
weight_metrics = MetricCollection({"Average": MeanMetric().to(DEVICE)})
_weighting_hook = make_weighting_hook(weight_metrics)


def _get_weighting(aggregator):
    """Return the sub-module that produces per-task weights, or None if unavailable."""
    if hasattr(aggregator, "gramian_weighting"):
        return aggregator.gramian_weighting
    if hasattr(aggregator, "weighting"):
        return aggregator.weighting
    return None


for _agg in (
    upgrad,
    mean,
    sum_,
    mgda,
    rgw,
    dualproj,
    pcgrad,
    imtlg,
    alignedm,
    nashmtl_32,
    cagrad_0_5,
):
    _w = _get_weighting(_agg)
    if _w is not None:
        _w.register_forward_hook(_weighting_hook)

# Gramian Weightings (for autogram engine)
upgrad_weighting = UPGradWeighting()
mean_weighting = MeanWeighting()
mgda_weighting = MGDAWeighting()
dualproj_weighting = DualProjWeighting()
pcgrad_weighting = PCGradWeighting()
imtlg_weighting = IMTLGWeighting()
cagrad_0_5_weighting = CAGradWeighting(c=0.5)
alignedm_weighting = AlignedMTLWeighting()
rgw_weighting = RandomWeighting()

for _w in (
    upgrad_weighting,
    mean_weighting,
    mgda_weighting,
    dualproj_weighting,
    pcgrad_weighting,
    imtlg_weighting,
    cagrad_0_5_weighting,
    alignedm_weighting,
    rgw_weighting,
):
    _w.register_forward_hook(_weighting_hook)

KEY_TO_GRAMIAN_WEIGHTING = {
    "UPGrad Mean": upgrad_weighting,
    "Mean": mean_weighting,
    "PCGrad": pcgrad_weighting,
    "MGDA": mgda_weighting,
    "DualProj Mean": dualproj_weighting,
    "IMTLG": imtlg_weighting,
    "CAGrad0.5": cagrad_0_5_weighting,
    "AlignedMTL Mean": alignedm_weighting,
    "Random": rgw_weighting,
}

KEY_TO_AGGREGATOR = {
    "UPGrad Mean": upgrad,
    "Mean": mean,
    "PCGrad": pcgrad,
    "MGDA": mgda,
    "DualProj Mean": dualproj,
    "IMTLG": imtlg,
    "CAGrad0.5": cagrad_0_5,
    "GradDrop": graddrop,
    "AlignedMTL Mean": alignedm,
    "NashMTL": nashmtl_32,
    "Random": rgw,
}
