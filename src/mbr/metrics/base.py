from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

import evaluate
import torch
from cachetools.func import fifo_cache
from datasets import Metric
from evaluate import EvaluationModule
from transformers import PreTrainedTokenizerBase
from transformers.utils import ModelOutput

from mbr import MBRConfig

MetricType = Union[Metric, EvaluationModule]


@dataclass
class MetricOutput(ModelOutput):
    """
    Args:
        scores (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            The metric scores for each sample (aggregated over all references).
        scores_per_reference (`torch.FloatTensor` of shape `(batch_size, num_samples, num_references)`):
            The pairwise metric scores for each sample and reference. `None` if the metric is computed corpus-level.
    """
    scores: torch.FloatTensor
    scores_per_reference: Optional[torch.FloatTensor] = None


class MetricRunner:
    """
    Applies the metric to samples and references (and optionally inputs) and calculates a metric score for each sample.
    This implementation uses the most basic approach, where samples and references are compared pairwise.
    Some metrics may support multi-reference evaluation or batching. Consider creating a subclass to make use of these
    features.
    """

    def __init__(self, mbr_config: MBRConfig, tokenizer: PreTrainedTokenizerBase):
        self.mbr_config = mbr_config
        # Ensure that mbr_config.metric_kwargs is hashable (because _compute_metric() uses lru_cache)
        if mbr_config.metric_kwargs:
            try:
                hash(self.mbr_config.metric_kwargs)
            except TypeError as e:
                raise TypeError(f"mbr_config.metric_kwargs must be hashable.") from e
        self.tokenizer = tokenizer
        self.metric = self._load_metric()
        self.metric_is_source_based = metric_is_source_based(self.metric)
        self._compute_metric_cached = fifo_cache(maxsize=self.mbr_config.metric_cache_size)(self._compute_metric)

    def _load_metric(self) -> MetricType:
        metric = self.mbr_config.metric
        if isinstance(metric, EvaluationModule):
            return metric
        elif isinstance(metric, str):
            metric = evaluate.load(metric, self.mbr_config.metric_config_name)
        else:
            raise ValueError(f"Invalid metric type: {type(metric)}")
        return metric

    def __call__(self,
                 input_ids: torch.LongTensor,
                 sample_ids: Tuple[torch.LongTensor],
                 reference_ids: Tuple[torch.LongTensor],
                 ) -> MetricOutput:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The input sequence ids.
            sample_ids (`tuple(torch.LongTensor)`):
                Tuple (one element for `num_samples`) of tensors of shape `(batch_size, sequence_length)` containing
                the sampled sequences.
            reference_ids:
                Tuple (one element for `num_references`) of tensors of shape `(batch_size, sequence_length)` containing
                the reference sequences.

        Returns:
            `MetricOutput` containing the metric scores.
        """

        # Detokenize
        if self.metric_is_source_based:
            str_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)  # shape: (batch_size,)
        else:
            str_inputs = None
        str_samples = []  # num_samples x batch_size
        for sample in sample_ids:
            str_samples.append(self.tokenizer.batch_decode(sample, skip_special_tokens=True))
        str_references = []  # num_references x batch_size
        for reference in reference_ids:
            str_references.append(self.tokenizer.batch_decode(reference, skip_special_tokens=True))

        if len(str_samples[0]) != len(str_references[0]):
            raise ValueError("Batch size of samples and references must match")
        if len(str_samples) != self.mbr_config.num_samples:
            raise ValueError("Number of samples must match `mbr_config.num_samples`")
        if len(str_references) != self.mbr_config.num_references:
            raise ValueError("Number of references must match `mbr_config.num_references`")

        # Compute metric
        scores_per_reference = self._compute_str_metric(str_samples, str_references, str_inputs)

        return MetricOutput(
            scores=scores_per_reference.mean(dim=-1),
            scores_per_reference=scores_per_reference,
        )

    def _compute_str_metric(self,
                            samples: List[List[str]],
                            references: List[List[str]],
                            inputs: List[str] = None,
                            ) -> torch.FloatTensor:
        batch_size = len(samples[0])
        metric_scores = torch.zeros((batch_size, self.mbr_config.num_samples, self.mbr_config.num_references))
        for i in range(batch_size):
            for j in range(self.mbr_config.num_samples):
                sample = samples[j][i]
                for k in range(self.mbr_config.num_references):
                    reference = references[k][i]
                    if inputs is not None:
                        score = self.compute_metric(
                            sources=(inputs[i],),
                            predictions=(sample,),
                            references=(reference,),
                            **self.mbr_config.metric_kwargs,
                        )
                    else:
                        score = self.compute_metric(
                            predictions=(sample,),
                            references=(reference,),
                            **self.mbr_config.metric_kwargs,
                        )
                    metric_scores[i, j, k] = score
        return metric_scores

    def _compute_metric(self, *args, **kwargs) -> float:
        # Call _compute() instead of compute() for performance reasons.
        # Since we are comparing individual samples, we do not need the overhead of compute().
        output = self.metric._compute(*args, **kwargs)
        if self.mbr_config.metric_output_field not in output:
            raise ValueError(f"Metric output does not contain '{self.mbr_config.metric_output_field}' "
                             f"Use `mbr_config.metric_output_field` to specify the correct field. "
                             f"Available fields: {list(output.keys())}"
                             )
        score = output[self.mbr_config.metric_output_field]
        if isinstance(score, list):
            score = score[0]
        return score

    def compute_metric(self, *args, **kwargs) -> float:
        return self._compute_metric_cached(*args, **kwargs)


def metric_is_source_based(metric: MetricType) -> bool:
    return "sources" in metric.features
