from pathlib import Path
from unittest import TestCase


class BeamSearchTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_beam_search(self):
        from experiments.reference_aggregation.baseline_beam_search import main
        out_path = main(self.testset, self.language_pair, limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(out_path.exists())
        self.assertIn(self.test_dir, out_path.parents)
        self.assertTrue(out_path.name.endswith(".de"))
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])
