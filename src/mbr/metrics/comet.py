import itertools
from typing import List, Tuple, Dict

import torch
from comet.models import RegressionMetric
from tqdm import tqdm

from mbr import MetricRunner


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
        self.batch_size_embed = batch_size_embed
        self.batch_size_estimate = batch_size_estimate
        self.progress_bar = progress_bar

    def _compute_str_metric(self,
                            samples: List[List[str]],
                            references: List[List[str]],
                            inputs: List[str] = None,
                            ) -> torch.FloatTensor:
        if inputs is None:
            raise NotImplementedError("CometMetricRunner requires source sequences (`inputs`) to be provided")
        batch_size = len(samples[0])
        metric_scores = torch.zeros((batch_size, self.mbr_config.num_samples, self.mbr_config.num_references))
        for i in tqdm(list(range(batch_size)), desc="comet", disable=not self.progress_bar):
            # Embed all sequences
            all_samples = [sample[i] for sample in samples]
            all_references = [reference[i] for reference in references]
            all_sequences = list(set(all_samples + all_references + inputs))
            all_encodings = self.comet.scorer.encoder.prepare_sample(all_sequences).to(self.comet.scorer.device)
            all_embeddings: Dict[str, torch.FloatTensor] = {}
            batches = itertools.zip_longest(range(0, len(all_sequences), self.batch_size_embed),
                                            range(self.batch_size_embed, len(all_sequences), self.batch_size_embed))
            for start_idx, end_idx in batches:
                embeddings = self.comet.scorer.get_sentence_embedding(
                    input_ids=all_encodings["input_ids"][start_idx:end_idx],
                    attention_mask=all_encodings["attention_mask"][start_idx:end_idx],
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                    all_embeddings[all_sequences[j]] = embeddings[j - start_idx]

            # Collect all input triples in a list
            input_triples: List[Tuple[str, str, str]] = []
            for j in range(self.mbr_config.num_samples):
                for k in range(self.mbr_config.num_references):
                    input_triples.append((inputs[i], samples[j][i], references[k][i]))
            input_triples = list(set(input_triples))  # deduplicate

            # Compute scores for all input triples
            input_triple_scores = {}
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
                    input_triple_scores[batch[j - start_idx]] = batch_scores.score[j - start_idx]

            for j in range(self.mbr_config.num_samples):
                for k in range(self.mbr_config.num_references):
                    metric_scores[i, j, k] = input_triple_scores[(inputs[i], samples[j][i], references[k][i])]

        return metric_scores
