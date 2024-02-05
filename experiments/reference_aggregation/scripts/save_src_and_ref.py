import argparse
from pathlib import Path

from experiments.reference_aggregation.experiment_utils import Testset


def main(testset: str, language_pair: str, limit_segments: int = None, out_dir: Path = None) -> (Path, Path):
    if out_dir is None:
        out_dir = Path(__file__).parent

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)

    dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

    src_out_path = translations_dir / f"{dataset}.src.{dataset.src_lang}"

    with open(src_out_path, "w") as f:
        for src in dataset.source_sentences:
            f.write(src + "\n")

    ref_out_path = translations_dir / f"{dataset}.ref.{dataset.tgt_lang}"
    with open(ref_out_path, "w") as f:
        for ref in dataset.references:
            f.write(ref + "\n")

    return src_out_path, ref_out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    src_path, ref_path = main(testset=args.testset, language_pair=args.language_pair,
        limit_segments=args.limit_segments, )
    print(f"Source sentences saved to {src_path}")
    print(f"References saved to {ref_path}")
