from pathlib import Path
from unittest import TestCase


class SaveSrcAndRefTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_save_src_and_ref(self):
        from experiments.reference_aggregation.scripts.save_src_and_ref import main
        src_path, ref_path = main(self.testset, self.language_pair, limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(src_path.exists())
        self.assertIn(self.test_dir, src_path.parents)
        self.assertTrue(src_path.name.endswith(".en"))
        self.assertTrue(ref_path.exists())
        self.assertIn(self.test_dir, ref_path.parents)
        self.assertTrue(ref_path.name.endswith(".de"))
        source_sentences = src_path.read_text().splitlines()
        self.assertEqual(len(source_sentences), 4)
        print(source_sentences[0])
        reference_sentences = ref_path.read_text().splitlines()
        self.assertEqual(len(reference_sentences), 4)
        print(reference_sentences[0])
