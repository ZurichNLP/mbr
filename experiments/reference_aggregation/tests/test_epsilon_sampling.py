from pathlib import Path
from unittest import TestCase

from experiments.reference_aggregation.fairseq_utils import load_model


class EpsilonSamplingTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_epsilon_sampling(self):
        model = load_model(self.language_pair)
        source_sentence = "This is a test."
        num_samples = 4
        # ε=0.02
        samples = model.sample(num_samples * [source_sentence], seed=42, sampling_epsilon_cutoff=0.02)
        self.assertEqual(len(samples), num_samples)
        self.assertIsInstance(samples[0], str)
        print(samples[0])
        # ε=0
        samples = model.sample(num_samples * [source_sentence], seed=42, sampling_epsilon_cutoff=0)
        self.assertEqual(len(samples), num_samples)
        self.assertIsInstance(samples[0], str)

    def test_extract_translations(self):
        # Generate samples
        from experiments.reference_aggregation.generate_samples import main as generate_samples
        jsonl_path = generate_samples(self.testset, self.language_pair, seed_no=0, num_samples=8, epsilon_cutoff=0.02,
                                      limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(jsonl_path.exists())
        # Extract
        from experiments.reference_aggregation.baseline_epsilon_sampling import main
        out_path = main(self.testset, self.language_pair, num_samples=8, epsilon_cutoff=0.02, seed_no=0,
                        out_dir=self.test_dir)
        self.assertTrue(out_path.exists())
        self.assertIn(self.test_dir, out_path.parents)
        self.assertTrue(out_path.name.endswith(".de"))
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])
