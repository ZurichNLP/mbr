from mbr import MBRConfig
from mbr.metrics.base import metric_is_source_based, MetricRunner


def load_metric_runner(mbr_config: MBRConfig, tokenizer=None) -> MetricRunner:
    if mbr_config.metric in ["fastchrf", "fastchrf-aggregate"]:
        from mbr.metrics.fastchrf import FastChrfMetricRunner
        return FastChrfMetricRunner(mbr_config, tokenizer)
    elif mbr_config.metric == "fastchrf-pairwise":
        from mbr.metrics.fastchrf import FastChrfMetricRunner
        return FastChrfMetricRunner(mbr_config, tokenizer, compute_pairwise_average=True)
    else:
        return MetricRunner(mbr_config, tokenizer)
