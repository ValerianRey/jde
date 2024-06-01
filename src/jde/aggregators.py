# Aggregators
from torchjd.aggregation import (
    AlignedMTLWrapper,
    CAGradWeighting,
    DualProjWrapper,
    GradDropAggregator,
    IMTLGWeighting,
    MeanWeighting,
    MGDAWeighting,
    NashMTLWeighting,
    PCGradWeighting,
    RandomWeighting,
    SumWeighting,
    UPGradWrapper,
    WeightedAggregator,
)
from torchmetrics import MeanMetric, MetricCollection

from jde.hooks import make_aggregator_hook, make_weighting_hook
from jde.metrics import CosineSimilarityToMatrixMean, MultiBatchWrapper
from jde.settings import DEVICE

upgrad = WeightedAggregator(UPGradWrapper(MeanWeighting()))
mean = WeightedAggregator(MeanWeighting())
sum_ = WeightedAggregator(SumWeighting())
mgda = WeightedAggregator(MGDAWeighting())
rgw = WeightedAggregator(RandomWeighting())
dualproj = WeightedAggregator(DualProjWrapper(MeanWeighting()))
pcgrad = WeightedAggregator(PCGradWeighting())
imtlg = WeightedAggregator(IMTLGWeighting())
graddrop = GradDropAggregator()
alignedm = WeightedAggregator(AlignedMTLWrapper(MeanWeighting()))
nashmtl_32 = WeightedAggregator(NashMTLWeighting(n_tasks=32, optim_niter=1))
cagrad_0_5 = WeightedAggregator(CAGradWeighting(c=0.5))

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

upgrad.weighting.register_forward_hook(_weighting_hook)
mean.weighting.register_forward_hook(_weighting_hook)
sum_.weighting.register_forward_hook(_weighting_hook)
mgda.weighting.register_forward_hook(_weighting_hook)
rgw.weighting.register_forward_hook(_weighting_hook)
dualproj.weighting.register_forward_hook(_weighting_hook)
pcgrad.weighting.register_forward_hook(_weighting_hook)
imtlg.weighting.register_forward_hook(_weighting_hook)
alignedm.weighting.register_forward_hook(_weighting_hook)
nashmtl_32.weighting.register_forward_hook(_weighting_hook)
cagrad_0_5.weighting.register_forward_hook(_weighting_hook)

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
