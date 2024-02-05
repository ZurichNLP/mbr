import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm


def main(testset: str, language_pair: str, num_samples: int, epsilon_cutoff: float, seed_no: int,
         out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    samples_dir = out_dir / "samples"
    assert samples_dir.exists()
    samples_path = samples_dir / f"samples.{testset}.{language_pair}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
    assert samples_path.exists()

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    out_path = translations_dir / f"{testset}.{language_pair}.epsilon{epsilon_cutoff}.seed{seed_no}.{language_pair.split('-')[1]}"

    with jsonlines.open(samples_path) as f_in, open(out_path, "w") as f_out:
        for line in tqdm(f_in):
            samples = line["samples"]
            assert len(samples) == num_samples
            f_out.write(samples[0] + "\n")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    args = parser.parse_args()

    out_path = main(testset=args.testset, language_pair=args.language_pair, num_samples=args.num_samples,
        epsilon_cutoff=args.epsilon_cutoff, seed_no=args.seed, )
    assert out_path.exists()
    print(f"Saved translations to {out_path}")
