import itertools
from typing import List, Tuple, Dict, Set

import torch
from cachetools import FIFOCache
from comet.models import RegressionMetric
from tqdm import tqdm

from mbr import MetricRunner, MetricOutput


class CometMetricRunner(MetricRunner):
    """
    Efficient usage of COMET for MBR, based on https://github.com/Unbabel/COMET

    The implementation is inspired by https://github.com/chanberg/COMET-mbr and
    https://github.com/Unbabel/COMET/blob/master/comet/cli/mbr.py
    """

    def __init__(self,
                 *args,
                 device=None,
                 batch_size_embed: int = 1,
                 batch_size_estimate: int = 1,
                 progress_bar: bool = False,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        if self.metric.__class__.__name__ != "COMET":
            raise ValueError(
                f"CometMetricRunner expects an evaluate.COMET metric, got {self.metric.__class__.__name__}")
        self.comet = self.metric
        if self.mbr_config.metric_output_field not in ["mean_score", "scores"]:
            raise ValueError(f"CometMetricRunner expects metric_output_field to be 'mean_score' or 'scores', "
                             f"got {self.mbr_config.metric_output_field}")
        if self.mbr_config.metric_kwargs:
            raise NotImplementedError("CometMetricRunner does not support metric_kwargs")
        if not isinstance(self.comet.scorer, RegressionMetric):
            raise NotImplementedError("CometMetricRunner only supports COMET models that are an instance of "
                                      "comet.models.RegressionMetric")
        if device is not None:
            self.comet.scorer = self.comet.scorer.to(device)
        self.comet.scorer.eval()
        self.batch_size_embed = batch_size_embed
        self.batch_size_estimate = batch_size_estimate
        self.progress_bar = progress_bar
        # We use a key-value cache, which is needed if the metric is called multiple times with similar inputs
        # (e.g. for MBR with iterative pruning).
        self.embedding_cache = FIFOCache(maxsize=self.mbr_config.metric_cache_size)
        self.score_cache = FIFOCache(maxsize=self.mbr_config.metric_cache_size)

    @torch.no_grad()
    def _compute_str_metric(self,
                            samples: List[List[str]],
                            references: List[List[str]],
                            inputs: List[str] = None,
                            ) -> torch.FloatTensor:
        if inputs is None:
            raise NotImplementedError("CometMetricRunner requires source sequences (`inputs`) to be provided")
        batch_size = len(samples[0])
        metric_scores = torch.zeros((batch_size, len(samples), len(references)))
        for i in tqdm(list(range(batch_size)), desc="comet", disable=not self.progress_bar):
            # Embed all sequences
            all_samples = [sample[i] for sample in samples]
            all_references = [reference[i] for reference in references]
            all_sequences = set(all_samples + all_references + inputs)

            all_embeddings: Dict[str, torch.FloatTensor] = {}
            # Populate embeddings from cache
            for sequence in list(all_sequences):
                if sequence in self.embedding_cache:
                    all_embeddings[sequence] = self.embedding_cache[sequence]
                    all_sequences.remove(sequence)

            # Compute embeddings for remaining sequences
            if all_sequences:
                all_sequences = list(all_sequences)
                encodings = self.comet.scorer.encoder.prepare_sample(all_sequences).to(self.comet.scorer.device)
                batches = itertools.zip_longest(range(0, len(all_sequences), self.batch_size_embed),
                                                range(self.batch_size_embed, len(all_sequences), self.batch_size_embed))
                for start_idx, end_idx in batches:
                    embeddings = self.comet.scorer.get_sentence_embedding(
                        input_ids=encodings["input_ids"][start_idx:end_idx],
                        attention_mask=encodings["attention_mask"][start_idx:end_idx],
                    )
                    for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                        embedding = embeddings[j - start_idx]
                        all_embeddings[all_sequences[j]] = embedding
                        self.embedding_cache[all_sequences[j]] = embedding

            # Collect all input triples in a list
            input_triples: Set[Tuple[str, str, str]] = set()
            for j in range(len(samples)):
                for k in range(len(references)):
                    input_triples.add((inputs[i], samples[j][i], references[k][i]))

            input_triple_scores: Dict[Tuple[str, str, str], torch.FloatTensor] = {}
            # Populate scores from cache
            for triple in list(input_triples):
                if triple in self.score_cache:
                    input_triple_scores[triple] = self.score_cache[triple]
                    input_triples.remove(triple)

            # Compute scores for remaining input triples
            input_triples: List = list(input_triples)
            batches = itertools.zip_longest(range(0, len(input_triples), self.batch_size_estimate),
                                            range(self.batch_size_estimate, len(input_triples),
                                                  self.batch_size_estimate))
            for start_idx, end_idx in batches:
                batch = input_triples[start_idx:end_idx]
                batch_scores = self.comet.scorer.estimate(
                    src_sentemb=torch.stack([all_embeddings[triple[0]] for triple in batch]),
                    mt_sentemb=torch.stack([all_embeddings[triple[1]] for triple in batch]),
                    ref_sentemb=torch.stack([all_embeddings[triple[2]] for triple in batch]),
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                    triple = batch[j - start_idx]
                    score = batch_scores.score[j - start_idx]
                    input_triple_scores[triple] = score
                    self.score_cache[triple] = score

            for j in range(len(samples)):
                for k in range(len(references)):
                    metric_scores[i, j, k] = input_triple_scores[(inputs[i], samples[j][i], references[k][i])]

        return metric_scores


class AggregateCometMetricRunner(CometMetricRunner):
    """
    Implements reference aggregation as described in "Linear-time Minimum Bayes Risk Decoding with Reference Aggregation"
     (Vamvas & Sennrich, 2024) https://arxiv.org/abs/2402.04251
    """

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

        # Compute metric
        scores = self._compute_str_metric(str_samples, str_references, str_inputs)

        return MetricOutput(
            scores=scores,
            scores_per_reference=None,
        )

    @torch.no_grad()
    def _compute_str_metric(self,
                            samples: List[List[str]],
                            references: List[List[str]],
                            inputs: List[str] = None,
                            ) -> torch.FloatTensor:
        if inputs is None:
            raise NotImplementedError("CometMetricRunner requires source sequences (`inputs`) to be provided")
        batch_size = len(samples[0])
        metric_scores = torch.zeros((batch_size, len(samples)))
        for i in tqdm(list(range(batch_size)), desc="comet", disable=not self.progress_bar):
            # Embed all sequences
            all_samples = [sample[i] for sample in samples]
            all_references = [reference[i] for reference in references]
            all_sequences = set(all_samples + all_references + inputs)

            all_embeddings: Dict[str, torch.FloatTensor] = {}
            # Populate embeddings from cache
            for sequence in list(all_sequences):
                if sequence in self.embedding_cache:
                    all_embeddings[sequence] = self.embedding_cache[sequence]
                    all_sequences.remove(sequence)

            # Compute embeddings for remaining sequences
            if all_sequences:
                all_sequences = list(all_sequences)
                encodings = self.comet.scorer.encoder.prepare_sample(all_sequences).to(self.comet.scorer.device)
                batches = itertools.zip_longest(range(0, len(all_sequences), self.batch_size_embed),
                                                range(self.batch_size_embed, len(all_sequences), self.batch_size_embed))
                for start_idx, end_idx in batches:
                    embeddings = self.comet.scorer.get_sentence_embedding(
                        input_ids=encodings["input_ids"][start_idx:end_idx],
                        attention_mask=encodings["attention_mask"][start_idx:end_idx],
                    )
                    for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                        embedding = embeddings[j - start_idx]
                        all_embeddings[all_sequences[j]] = embedding
                        self.embedding_cache[all_sequences[j]] = embedding

            # Compute average reference embedding
            avg_reference_embedding = torch.stack([all_embeddings[reference] for reference in all_references]).mean(dim=0)

            # Collect all input triples in a list
            input_triples: Set[Tuple[str, str, str]] = set()
            for j in range(len(samples)):
                input_triples.add((inputs[i], samples[j][i], "avg"))

            input_triple_scores: Dict[Tuple[str, str, str], torch.FloatTensor] = {}
            # Populate scores from cache
            for triple in list(input_triples):
                if triple in self.score_cache:
                    input_triple_scores[triple] = self.score_cache[triple]
                    input_triples.remove(triple)

            # Compute scores for remaining input triples
            input_triples: List = list(input_triples)
            batches = itertools.zip_longest(range(0, len(input_triples), self.batch_size_estimate),
                                            range(self.batch_size_estimate, len(input_triples),
                                                  self.batch_size_estimate))
            for start_idx, end_idx in batches:
                batch = input_triples[start_idx:end_idx]
                batch_scores = self.comet.scorer.estimate(
                    src_sentemb=torch.stack([all_embeddings[triple[0]] for triple in batch]),
                    mt_sentemb=torch.stack([all_embeddings[triple[1]] for triple in batch]),
                    ref_sentemb=avg_reference_embedding.unsqueeze(0).repeat(len(batch), 1),
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                    triple = batch[j - start_idx]
                    score = batch_scores.score[j - start_idx]
                    input_triple_scores[triple] = score
                    self.score_cache[triple] = score

            for j in range(len(samples)):
                metric_scores[i, j] = input_triple_scores[(inputs[i], samples[j][i], "avg")]

        return metric_scores
