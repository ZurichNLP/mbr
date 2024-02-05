import argparse
import math
from pathlib import Path
from typing import List

import jsonlines
from tqdm import tqdm

from experiments.reference_aggregation.experiment_utils import Testset
from experiments.reference_aggregation.mbr_utils import load_utility


def main(testset: str, language_pair: str, seed_no: int, utility_name: str, chrf_eps_smoothing: bool = False,
         topk: int = 20, num_samples: int = 1024, epsilon_cutoff: float = 0.02, limit_segments: int = None,
         out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

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

    # s = n/1, n/2, n/4, n/8, ..., n/n
    s_values = [int(num_samples / 2 ** i) for i in range(int(math.log2(num_samples)) + 1)]
    assert s_values[0] == num_samples
    assert s_values[-1] == 1

    utility = load_utility(utility_name)

    if utility_name == "chrf" and chrf_eps_smoothing:
        utility.eps_smoothing = True

    # Compute rankings for n-by-s and aggregate, for each s
    n_by_s_rankings: List[List[List[int]]] = []  # segments x s_values x topk
    aggregate_rankings: List[List[List[int]]] = []  # segments x s_values x topk
    for i in tqdm(list(range(len(dataset.source_sentences))), desc="segments"):

        # For COMET: compute embeddings
        if hasattr(utility, "compute_features"):
            utility.clear_features()
            input_sequences = {dataset.source_sentences[i]} | set(samples[i]) | set(references[i])
            utility.compute_features(input_sequences)

        n_by_s_rankings.append([])
        for s in s_values:
            n_by_s_ranking = utility.rank_samples_n_by_s(dataset.source_sentences[i], samples[i], references[i], s=s)
            n_by_s_ranking = n_by_s_ranking[:topk]
            n_by_s_rankings[-1].append(n_by_s_ranking.tolist())
        aggregate_rankings.append([])
        for s in s_values:
            aggregate_ranking = utility.rank_samples_aggregate(dataset.source_sentences[i], samples[i], references[i],
                                                               s=s)
            aggregate_ranking = aggregate_ranking[:topk]
            aggregate_rankings[-1].append(aggregate_ranking.tolist())

    # Save top-k rankings to jsonl file
    output_dir = out_dir / "validation_output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{utility_name}{'-eps' if chrf_eps_smoothing else ''}.top{topk}.jsonl"
    with jsonlines.open(output_path, mode="w") as f:
        for i, s in enumerate(s_values):
            f.write({"method": "n_by_s", "s": s, "rankings": [ranking[i] for ranking in n_by_s_rankings]})
        for i, s in enumerate(s_values):
            f.write({"method": "aggregate", "s": s, "rankings": [ranking[i] for ranking in aggregate_rankings]})

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    translations_prefix = f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{utility_name}{'-eps' if chrf_eps_smoothing else ''}"

    # Save top-1 translations for n-by-s
    for j, s in enumerate(s_values):
        n_by_s_translations_path = translations_dir / f"{translations_prefix}.n_by_s.s{s}.{dataset.tgt_lang}"
        with open(n_by_s_translations_path, "w") as f:
            for i, rankings in enumerate(n_by_s_rankings):
                ranking = rankings[j]
                f.write(samples[i][ranking[0]] + "\n")

    # Save top-1 translations for aggregate
    for j, s in enumerate(s_values):
        aggregate_translations_path = translations_dir / f"{translations_prefix}.aggregate.s{s}.{dataset.tgt_lang}"
        with open(aggregate_translations_path, "w") as f:
            for i, rankings in enumerate(aggregate_rankings):
                ranking = rankings[j]
                f.write(samples[i][ranking[0]] + "\n")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--chrf-eps-smoothing', action='store_true',
                        help='Use epsilon smoothing for ChrF (default: False = effective order smoothing)')
    parser.add_argument('--topk', type=int, default=20, help='Number of top translations to save in the jsonl file')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    jsonl_path = main(testset=args.testset, language_pair=args.language_pair, seed_no=args.seed,
                      utility_name=args.utility, chrf_eps_smoothing=args.chrf_eps_smoothing, topk=args.topk,
                      num_samples=args.num_samples, epsilon_cutoff=args.epsilon_cutoff,
                      limit_segments=args.limit_segments, )
    print(f"Saved results file to {jsonl_path}")
