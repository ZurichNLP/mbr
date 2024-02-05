from unittest import TestCase

from experiments.reference_aggregation.mbr_utils import ChrfUtility


class CometTestCase(TestCase):

    def setUp(self):
        self.chrf = ChrfUtility()

    def test_rank_samples_n_by_n(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.",
                   "Dies ist ein Test."]
        references = samples

        indices = self.chrf.rank_samples_n_by_s(source_sequence, samples, references, s=4)
        self.assertEqual(len(samples), len(indices))
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.chrf.rank_samples_n_by_s(source_sequence, samples[::-1], references, s=4)
        self.assertListEqual([2, 3, 0, 1], indices.tolist())

        # Test reference order invariance
        indices = self.chrf.rank_samples_n_by_s(source_sequence, samples, references[::-1], s=4)
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

    def test_rank_samples_n_by_1(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.",
                   "Dies ist ein Test."]
        references = samples

        indices = self.chrf.rank_samples_n_by_s(source_sequence, samples, references, s=1)
        self.assertEqual(0, indices[0])  # Perfect match with itself
        self.assertListEqual([0, 1, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.chrf.rank_samples_n_by_s(source_sequence, samples[::-1], references, s=1)
        self.assertListEqual([3, 2, 0, 1], indices.tolist())

    def test_rank_samples_aggregate(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.",
                   "Dies ist ein Test."]
        references = samples

        indices = self.chrf.rank_samples_aggregate(source_sequence, samples, references, s=1)
        self.assertEqual(len(samples), len(indices))
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.chrf.rank_samples_aggregate(source_sequence, samples[::-1], references, s=1)
        self.assertListEqual([2, 3, 0, 1], indices.tolist())

        # Test reference order invariance
        indices = self.chrf.rank_samples_aggregate(source_sequence, samples, references[::-1], s=1)
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

    def test_rank_samples_aggregate_partial(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.",
                   "Dies ist ein Test."]
        references = samples

        indices = self.chrf.rank_samples_aggregate(source_sequence, samples, references, s=2)
        self.assertEqual(len(samples), len(indices))
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.chrf.rank_samples_aggregate(source_sequence, samples[::-1], references, s=2)
        self.assertListEqual([2, 3, 0, 1], indices.tolist())

        # Test (partial) reference order invariance: change order of references within partitions
        indices = self.chrf.rank_samples_aggregate(source_sequence, samples,
                                                   references[:2][::-1] + references[2:][::-1], s=2)
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

    def test_rank_samples_disaggregated_is_equivalent_to_n_by_n(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.",
                   "Dies ist ein Test."]
        references = samples

        n_by_n_indices = self.chrf.rank_samples_n_by_s(source_sequence, samples, references, s=4)
        aggregate_indices = self.chrf.rank_samples_aggregate(source_sequence, samples, references, s=4)
        self.assertListEqual(n_by_n_indices.tolist(), aggregate_indices.tolist())
