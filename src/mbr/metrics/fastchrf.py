from typing import List, Tuple

import torch
from fastchrf import pairwise_chrf, aggregate_chrf
from transformers import PreTrainedTokenizerBase

from mbr import MetricRunner, MBRConfig, MetricOutput


class FastChrfMetricRunner(MetricRunner):
    """
    MetricRunner for fastChrF. See https://github.com/jvamvas/fastChrF for more information.
    
    Args:
        mbr_config
        tokenizer
        compute_pairwise_average: Default: False. If True, use fastchr.chrf_pairwise() to calculate exact ChrF scores
            for each sample-reference pair and then average them; this corresponds to a fast implementation of the
            original ChrF metric. If False, use fastchr.chrf_aggregate() to directly calculate aggregate fastChrF scores
            across all references; note that the result will be different from the original ChrF metric.
    """

    def __init__(self,
                 mbr_config: MBRConfig,
                 tokenizer: PreTrainedTokenizerBase,
                 compute_pairwise_average: bool = False,
                 ):
        self.mbr_config = mbr_config
        self.tokenizer = tokenizer
        self.metric_is_source_based = False
        self.char_order = mbr_config.metric_kwargs.get("char_order", 6)
        self.beta = mbr_config.metric_kwargs.get("beta", 2)
        self.remove_whitespace = mbr_config.metric_kwargs.get("remove_whitespace", True)
        self.eps_smoothing = mbr_config.metric_kwargs.get("eps_smoothing", False)
        self.compute_pairwise_average = compute_pairwise_average
        
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

        # Transpose to batch_size x num_samples/num_references
        str_samples = list(zip(*str_samples))
        str_references = list(zip(*str_references))

        if self.compute_pairwise_average:
            output = self._compute_pairwise_chrf(str_samples, str_references)
        else:
            output = self._compute_aggregate_chrf(str_samples, str_references)
        return output

    def _compute_pairwise_chrf(self, samples: List[List[str]], references: List[List[str]]) -> MetricOutput:
        scores_per_reference = pairwise_chrf(
            samples,
            references,
            char_order=self.char_order,
            beta=self.beta,
            remove_whitespace=self.remove_whitespace,
            eps_smoothing=self.eps_smoothing,
        )
        scores_per_reference = torch.tensor(scores_per_reference)
        scores = scores_per_reference.mean(dim=-1)
        return MetricOutput(
            scores=scores,
            scores_per_reference=scores_per_reference,
        )

    def _compute_aggregate_chrf(self, samples: List[List[str]], references: List[List[str]]) -> MetricOutput:
        scores = aggregate_chrf(
            samples,
            references,
            char_order=self.char_order,
            beta=self.beta,
            remove_whitespace=self.remove_whitespace,
            eps_smoothing=self.eps_smoothing,
        )
        scores = torch.tensor(scores)
        return MetricOutput(
            scores=scores,
            scores_per_reference=None,
        )
