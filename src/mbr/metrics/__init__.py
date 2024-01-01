from mbr import MBRConfig
from mbr.metrics.base import metric_is_source_based, MetricRunner


def load_metric_runner(mbr_config: MBRConfig, tokenizer=None) -> MetricRunner:
    if mbr_config.metric == "fastchrf":
        from mbr.metrics.fastchrf import FastChrfMetricRunner
        return FastChrfMetricRunner(mbr_config, tokenizer)
    else:
        return MetricRunner(mbr_config, tokenizer)
