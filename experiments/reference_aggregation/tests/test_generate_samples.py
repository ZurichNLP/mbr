from pathlib import Path
from unittest import TestCase

import jsonlines


class GenerateSamplesTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_generate_samples(self):
        from experiments.reference_aggregation.generate_samples import main
        out_path = main(self.testset, self.language_pair, seed_no=0, num_samples=8, epsilon_cutoff=0.02,
                        limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(out_path.exists())
        self.assertIn(self.test_dir, out_path.parents)
        self.assertTrue(out_path.name.endswith(".jsonl"))
        with jsonlines.open(out_path) as f:
            data = list(f)
        self.assertEqual(len(data), 4)
        for line in data:
            self.assertIn("samples", line)
            self.assertEqual(len(line["samples"]), 8)
            self.assertIsInstance(line["samples"][0], str)
        print(data[0]["samples"][0])
