import itertools
import logging
from collections import namedtuple
from typing import Set, Dict, List, Tuple

import evaluate
import numpy as np
import torch
from fastchrf import pairwise_chrf, aggregate_chrf


class ChrfUtility:

    def __init__(self, char_order: int = 6, beta: float = 2.0, remove_whitespace: bool = True, eps_smoothing: bool = False):
        self.char_order = char_order
        self.beta = beta
        self.remove_whitespace = remove_whitespace
        self.eps_smoothing = eps_smoothing

    def rank_samples_n_by_s(self, source_sequence: str, samples: List[str], references: List[str], s: int = None) -> np.ndarray:
        """
        Returns the indices of the samples sorted by their utility score, in descending order.
        :param s: The number of references to subsample from the list of references (default: all references)
        """
        if s is None:
            s = len(references)
        assert s <= len(references)
        references = references[:s]

        metric_scores = pairwise_chrf(
            [samples], [references],
            char_order=self.char_order,
            beta=self.beta,
            remove_whitespace=self.remove_whitespace,
            eps_smoothing=self.eps_smoothing,
        )[0]
        metric_scores = np.array(metric_scores)  # num_samples x s

        # Sort the samples by their average score
        sample_scores = metric_scores.mean(axis=1)
        sample_indices = sample_scores.argsort()[::-1]
        return sample_indices

    def rank_samples_aggregate(self, source_sequence: str, samples: List[str], references: List[str], s: int) -> np.ndarray:
        """
        Returns the indices of the samples sorted by their utility score, in descending order.
        :param s: The number of aggregate references
        """
        assert s <= len(references)

        num_partitions = s
        partition_size = len(references) // num_partitions
        reference_partitions = [references[i * partition_size:(i + 1) * partition_size] for i in range(num_partitions)]

        metric_scores = aggregate_chrf(
            num_partitions * [samples], reference_partitions,
            char_order=self.char_order,
            beta=self.beta,
            remove_whitespace=self.remove_whitespace,
            eps_smoothing=self.eps_smoothing,
        )
        metric_scores = np.array(metric_scores).transpose()  # num_samples x s

        # Sort the samples by their average score
        sample_scores = metric_scores.mean(axis=1)
        sample_indices = sample_scores.argsort()[::-1]
        return sample_indices


CometInputTriple = namedtuple("CometInputTriple", ["src", "hyp", "ref"])


class CometUtility:

    def __init__(self,
                 model_name: str,
                 batch_size_embed: int = 1,
                 batch_size_estimate: int = 1,
                 ):
        self.model_name = model_name
        self.batch_size_embed = batch_size_embed
        self.batch_size_estimate = batch_size_estimate
        self.scorer = evaluate.load("comet", model_name).scorer
        if torch.cuda.is_available():
            self.scorer = self.scorer.to("cuda:0")
        else:
            logging.warning("CUDA not available, using CPU")
        self.scorer.eval()
        self.device = self.scorer.device
        self.embeddings: Dict[str, torch.FloatTensor] = {}

    @torch.no_grad()
    def compute_features(self, input_sequences: Set[str]):
        assert not self.scorer.training
        input_sequences = list(input_sequences)
        encodings = self.scorer.encoder.prepare_sample(input_sequences).to(self.device)
        batches = itertools.zip_longest(range(0, len(input_sequences), self.batch_size_embed),
                                        range(self.batch_size_embed, len(input_sequences), self.batch_size_embed))
        for start_idx, end_idx in batches:
            embeddings = self.scorer.get_sentence_embedding(
                input_ids=encodings["input_ids"][start_idx:end_idx],
                attention_mask=encodings["attention_mask"][start_idx:end_idx],
            )
            for j in range(start_idx, end_idx if end_idx is not None else len(input_sequences)):
                embedding = embeddings[j - start_idx]
                self.embeddings[input_sequences[j]] = embedding

    def clear_features(self):
        self.embeddings = {}

    @torch.no_grad()
    def rank_samples_n_by_s(self, source_sequence: str, samples: List[str], references: List[str], s: int = None) -> np.ndarray:
        """
        Returns the indices of the samples sorted by their utility score, in descending order.
        :param s: The number of references to subsample from the list of references (default: all references)
        """
        if s is None:
            s = len(references)
        assert s <= len(references)
        references = references[:s]
        assert not self.scorer.training

        # Collect all unique input triples
        input_triples: Set[Tuple[str, str, str]] = set()
        for sample in samples:
            for reference in references:
                input_triples.add(CometInputTriple(src=source_sequence, hyp=sample, ref=reference))
        input_triples: List = list(input_triples)

        # Compute scores for all input triples
        triple_scores: Dict[CometInputTriple, torch.tensor] = {}
        batches = itertools.zip_longest(range(0, len(input_triples), self.batch_size_estimate),
                                        range(self.batch_size_estimate, len(input_triples), self.batch_size_estimate))
        for start_idx, end_idx in batches:
            batch = input_triples[start_idx:end_idx]
            batch_scores = self.scorer.estimate(
                src_sentemb=torch.stack([self.embeddings[input.src] for input in batch]),
                mt_sentemb=torch.stack([self.embeddings[input.hyp] for input in batch]),
                ref_sentemb=torch.stack([self.embeddings[input.ref] for input in batch]),
            )
            for i in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                triple = batch[i - start_idx]
                score = batch_scores.score[i - start_idx]
                triple_scores[triple] = score

        # Fill in the metric scores matrix
        metric_scores = torch.zeros((len(samples), len(references)))
        for i, sample in enumerate(samples):
            for j, reference in enumerate(references):
                metric_scores[i, j] = triple_scores[CometInputTriple(src=source_sequence, hyp=sample, ref=reference)]

        # Sort the samples by their average score
        sample_scores = metric_scores.mean(dim=1)
        sample_indices = sample_scores.argsort(descending=True)
        return sample_indices.cpu().numpy()

    @torch.no_grad()
    def rank_samples_aggregate(self, source_sequence: str, samples: List[str], references: List[str], s: int) -> np.ndarray:
        """
        Returns the indices of the samples sorted by their utility score, in descending order.
        :param s: The number of aggregate referencesq
        """
        assert s <= len(references)
        assert not self.scorer.training

        num_partitions = s
        partition_size = len(references) // num_partitions

        # Add aggregate reference embeddings to the embeddings cache
        reference_embeddings = torch.stack([self.embeddings[reference] for reference in references])
        avg_reference_embeddings = reference_embeddings.view(num_partitions, partition_size, -1).mean(dim=1)
        for partition_id in range(num_partitions):
            self.embeddings[f"aggregate_{partition_id}"] = avg_reference_embeddings[partition_id]

        # Collect all unique input triples
        input_triples: Set[Tuple[str, str, str]] = set()
        for sample in samples:
            for partition_id in range(s):
                input_triples.add(CometInputTriple(src=source_sequence, hyp=sample, ref=f"aggregate_{partition_id}"))
        input_triples: List = list(input_triples)

        # Compute scores for all input triples
        triple_scores: Dict[CometInputTriple, torch.tensor] = {}
        batches = itertools.zip_longest(range(0, len(input_triples), self.batch_size_estimate),
                                        range(self.batch_size_estimate, len(input_triples), self.batch_size_estimate))
        for start_idx, end_idx in batches:
            batch = input_triples[start_idx:end_idx]
            batch_scores = self.scorer.estimate(
                src_sentemb=torch.stack([self.embeddings[input.src] for input in batch]),
                mt_sentemb=torch.stack([self.embeddings[input.hyp] for input in batch]),
                ref_sentemb=torch.stack([self.embeddings[input.ref] for input in batch]),
            )
            for i in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                triple = batch[i - start_idx]
                score = batch_scores.score[i - start_idx]
                triple_scores[triple] = score

        # Fill in the metric scores matrix
        metric_scores = torch.zeros((len(samples), num_partitions))
        for i, sample in enumerate(samples):
            for partition_id in range(s):
                metric_scores[i, partition_id] = triple_scores[CometInputTriple(src=source_sequence, hyp=sample, ref=f"aggregate_{partition_id}")]

        # Sort the samples by their average score
        sample_scores = metric_scores.mean(dim=1)
        sample_indices = sample_scores.argsort(descending=True)
        return sample_indices.cpu().numpy()


def load_utility(utility_name: str):
    if utility_name == "chrf":
        return ChrfUtility()
    elif utility_name.startswith("comet22"):
        return CometUtility("Unbabel/wmt22-comet-da", batch_size_embed=128, batch_size_estimate=128)
    elif utility_name.startswith("cometinho"):
        return CometUtility("eamt22-cometinho-da", batch_size_embed=512, batch_size_estimate=512)
    else:
        raise ValueError(f"Unknown utility {utility_name}")
