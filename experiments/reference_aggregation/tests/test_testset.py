from unittest import TestCase

from experiments.reference_aggregation.experiment_utils import Testset


class TestsetTestCase(TestCase):

    def setUp(self):
        self.testsets = ["wmt21", "wmt22"]
        self.language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]

    def test_load_testsets(self):
        for testset in self.testsets:
            for language_pair in self.language_pairs:
                data = Testset.from_wmt(testset, language_pair)
                self.assertEqual(language_pair, f"{data.src_lang}-{data.tgt_lang}")
                self.assertEqual(len(data.source_sentences), len(data.references))
