import argparse
from pathlib import Path
from typing import List, Tuple

import jsonlines

from experiments.reference_aggregation.experiment_utils import Testset


def main(testset: str, language_pair: str, seed_no: int, fine_utility_name: str, topk: int, accuracy_topk: int,
         method: str, num_samples: int = 1024, epsilon_cutoff: float = 0.02, coarse_utility_name: str = None,
         limit_segments: int = None, out_dir: Path = None) -> List[Tuple[int, float]]:
    """
    Returns a series of (s, accuracy) tuples, starting with the highest s
    """
    if out_dir is None:
        out_dir = Path(__file__).parent

    if coarse_utility_name is None:
        coarse_utility_name = fine_utility_name

    assert topk <= num_samples
    assert accuracy_topk <= topk

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

    output_dir = out_dir / "validation_output"
    assert output_dir.exists()
    fine_output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{fine_utility_name}.top{topk}.jsonl"
    with jsonlines.open(fine_output_path) as f:
        fine_data = list(f)
    coarse_output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{coarse_utility_name}.top{topk}.jsonl"
    with jsonlines.open(coarse_output_path) as f:
        coarse_data = list(f)

    # Get n-by-n top-1 samples – should not matter which method
    n_by_n_lines = [line for line in fine_data if line["s"] == num_samples]
    assert len(n_by_n_lines) == 2
    for ranking in zip(n_by_n_lines[0]["rankings"], n_by_n_lines[1]["rankings"]):
        assert ranking[0] == ranking[1]
    n_by_n_rankings = n_by_n_lines[0]["rankings"]
    n_by_n_top1_samples = [samples[i][n_by_n_rankings[i][0]].strip() for i in range(len(samples))]

    # Get top-k accuracies for efficiency method
    method_lines = [line for line in coarse_data if line["method"] == method]
    assert len(method_lines) == len(coarse_data) / 2
    s_values = list(sorted([line["s"] for line in method_lines], reverse=True))
    accuracies = []  # for each s
    for s in s_values:
        s_lines = [line for line in method_lines if line["s"] == s]
        assert len(s_lines) == 1
        s_rankings = s_lines[0]["rankings"]
        s_topk_samples = [{samples[i][ranking].strip() for ranking in s_rankings[i][:accuracy_topk]} for i in
                          range(len(samples))]
        s_num_correct = sum([1 if n_by_n_top1_samples[i] in s_topk_samples[i] else 0 for i in range(len(samples))])
        s_accuracy = s_num_correct / len(samples)
        accuracies.append(s_accuracy)

    # Format: (1,-0.4)(2,-0.6)(4,-0.5)(8,0.1)(16,0.1)(32,0.2)(64,0.1)(128,-0.0)(256,-0.0)
    series = [(s, accuracy) for s, accuracy in zip(s_values, accuracies)]
    return series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--coarse-utility', choices=['chrf', 'cometinho', 'comet22'], default=None,
                        help='Utility used for coarse-grained method (default: same as fine-grained)')
    parser.add_argument('--topk', type=int, default=20,
                        help='Number of top translations that have been saved in the jsonl file')
    parser.add_argument('--method', choices=['n_by_s', 'aggregate'], required=True)
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--accuracy-topk', type=int, default=None,
                        help='Number of top translations that are used to compute the accuracy (default: same as data-topk)')
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    if args.coarse_utility is None:
        args.coarse_utility = args.utility
    if args.accuracy_topk is None:
        args.accuracy_topk = args.topk

    series = main(testset=args.testset, language_pair=args.language_pair, seed_no=args.seed,
        fine_utility_name=args.utility, coarse_utility_name=args.coarse_utility, topk=args.topk, method=args.method,
        num_samples=args.num_samples, epsilon_cutoff=args.epsilon_cutoff, accuracy_topk=args.accuracy_topk,
        limit_segments=args.limit_segments, )

    # Format: (1,-0.4)(2,-0.6)(4,-0.5)(8,0.1)(16,0.1)(32,0.2)(64,0.1)(128,-0.0)(256,-0.0)
    series_str = "".join([f"({s},{accuracy:.5f})" for s, accuracy in series])
    print(
        f"Testset: {args.testset}, language pair: {args.language_pair}, seed: {args.seed}, fine utility: {args.utility}, coarse utility: {args.coarse_utility}, topk: {args.topk}, method: {args.method}")
    print(f"Top-{args.accuracy_topk} accuracy:")
    print(series_str)
    print()
