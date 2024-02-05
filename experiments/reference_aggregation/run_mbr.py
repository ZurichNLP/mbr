import argparse
import time
from pathlib import Path
from typing import List, Optional

import jsonlines
from tqdm import tqdm

from experiments.reference_aggregation.experiment_utils import Testset
from experiments.reference_aggregation.mbr_utils import load_utility


def main(method: str, topk: Optional[int], testset: str, language_pair: str, seed_no: int, fine_utility_name: str,
         num_samples: int = 1024, epsilon_cutoff: float = 0.02, coarse_utility_name: str = None,
         limit_segments: int = None, log_time: bool = False, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    if coarse_utility_name is None:
        coarse_utility_name = fine_utility_name

    if method in {'aggregate_to_fine', 'coarse_to_fine'}:
        assert topk <= num_samples

    dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

    samples_dir = out_dir / "samples"
    assert samples_dir.exists()
    samples_path = samples_dir / f"samples.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
    assert samples_path.exists()
    with jsonlines.open(samples_path) as f:
        samples = [line["samples"] for line in f]
    samples = [sample[:num_samples] for sample in samples]
    if limit_segments is not None:
        samples = samples[:limit_segments]

    assert len(samples) == len(dataset.source_sentences)
    assert all(len(sample) == num_samples for sample in samples)

    references = samples

    utility = load_utility(fine_utility_name)
    if coarse_utility_name == fine_utility_name:
        coarse_utility = utility
    else:
        coarse_utility = load_utility(coarse_utility_name)

    translations: List[str] = []

    if log_time:
        start_time = time.time()

    for i in tqdm(list(range(len(dataset.source_sentences))), desc="segments"):

        # For COMET: compute embeddings
        if hasattr(coarse_utility, "compute_features"):
            coarse_utility.clear_features()
            input_sequences = {dataset.source_sentences[i]} | set(samples[i]) | set(references[i])
            coarse_utility.compute_features(input_sequences)

        if method == 'pairwise':
            n_by_n_ranking = utility.rank_samples_n_by_s(dataset.source_sentences[i], samples[i], references[i],
                                                         s=num_samples)
            translation = samples[i][n_by_n_ranking[0]]
        elif method == 'aggregate':
            aggregate_ranking = utility.rank_samples_aggregate(dataset.source_sentences[i], samples[i], references[i],
                                                               s=1)
            translation = samples[i][aggregate_ranking[0]]
        elif method == 'aggregate_to_fine':
            aggregate_ranking = coarse_utility.rank_samples_aggregate(dataset.source_sentences[i], samples[i],
                                                                      references[i], s=1)
            topk_samples = [samples[i][aggregate_ranking[j]] for j in range(topk)]

            if fine_utility_name != coarse_utility_name and hasattr(utility, "compute_features"):
                utility.clear_features()
                input_sequences = {dataset.source_sentences[i]} | set(topk_samples) | set(references[i])
                utility.compute_features(input_sequences)

            fine_ranking = utility.rank_samples_n_by_s(dataset.source_sentences[i], topk_samples, references[i],
                                                       s=num_samples)
            translation = topk_samples[fine_ranking[0]]
        elif method == 'coarse_to_fine':
            coarse_ranking = coarse_utility.rank_samples_n_by_s(dataset.source_sentences[i], samples[i], references[i],
                                                                s=num_samples)
            topk_samples = [samples[i][coarse_ranking[j]] for j in range(topk)]

            if fine_utility_name != coarse_utility_name and hasattr(utility, "compute_features"):
                utility.clear_features()
                input_sequences = {dataset.source_sentences[i]} | set(topk_samples) | set(references[i])
                utility.compute_features(input_sequences)

            fine_ranking = utility.rank_samples_n_by_s(dataset.source_sentences[i], topk_samples, references[i],
                                                       s=num_samples)
            translation = topk_samples[fine_ranking[0]]
        else:
            raise ValueError(f"Unknown method: {method}")
        translations.append(translation)

    if log_time:
        print(f"Average time per segment: {(time.time() - start_time) / len(dataset.source_sentences):.5f} seconds")

    assert len(translations) == len(dataset.source_sentences)

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    out_path = translations_dir / f"mbr.{dataset}.{method}{'.top' + str(topk) if method in {'aggregate_to_fine', 'coarse_to_fine'} else ''}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{coarse_utility_name + '-to-' if coarse_utility_name != fine_utility_name else ''}{fine_utility_name}.{dataset.tgt_lang}"
    with open(out_path, "w") as f:
        for translation in translations:
            f.write(translation + "\n")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['pairwise', 'aggregate', 'aggregate_to_fine', 'coarse_to_fine'],
                        required=True)
    parser.add_argument('--topk', type=int, default=20,
                        help='Number of samples to prune to in aggregate_to_fine method')
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--coarse-utility', choices=['chrf', 'cometinho', 'comet22'], default=None,
                        help='Utility used for coarse-grained method (default: same as fine-grained)')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    parser.add_argument('--log-time', action='store_true',
                        help='Print average wall-clock time per segment (used for benchmarking)')
    args = parser.parse_args()

    if args.coarse_utility is None:
        args.coarse_utility = args.utility

    out_path = main(method=args.method, topk=args.topk, testset=args.testset, language_pair=args.language_pair,
        seed_no=args.seed, fine_utility_name=args.utility, coarse_utility_name=args.coarse_utility,
        num_samples=args.num_samples, epsilon_cutoff=args.epsilon_cutoff, limit_segments=args.limit_segments,
        log_time=args.log_time, )
    assert out_path.exists()
    print(f"Saved translations to {out_path}")
