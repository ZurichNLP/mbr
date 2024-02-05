import os
import unittest
from unittest import TestCase

import evaluate
import torch
from transformers import AutoTokenizer

from mbr import MetricRunner, MBRConfig
from mbr.metrics import metric_is_source_based


class MetricUtilsTestCase(TestCase):

    def setUp(self):
        self.mbr_config = MBRConfig(
            metric="chrf",
            metric_output_field="score",
            num_samples=3,
            num_references=2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        self.inputs = [  # shape: (batch_size,)
            "This is an input sentence.",
            "This is another input sentence.",
        ]
        self.samples = [  # num_samples x batch_size
            ["This is a sample sentence.", "Something totally different."],
            ["This is a sample sentence.", "This a third sample sentence."],
            ["Something totally different.", "This is a fourth sample sentence."],
        ]
        self.references = [  # num_references x batch_size
            ["This is a reference sentence.", "This is another reference sentence."],
            ["This is a reference sentence.", "This is a fourth reference sentence."],
        ]
        self.input_ids = self.tokenizer(self.inputs, return_tensors="pt", padding=True).input_ids
        self.sample_ids = tuple([self.tokenizer(sample, return_tensors="pt", padding=True).input_ids for sample in self.samples])
        self.reference_ids = tuple([self.tokenizer(reference, return_tensors="pt", padding=True).input_ids for reference in self.references])

    def test_is_source_based__chrf(self):
        chrf = evaluate.load("chrf")
        self.assertFalse(metric_is_source_based(chrf))

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_is_source_based__comet(self):
        comet = evaluate.load("comet", "eamt22-cometinho-da")
        self.assertTrue(metric_is_source_based(comet))

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_is_source_based__bleurt(self):
        bleurt = evaluate.load("bleurt")
        self.assertFalse(metric_is_source_based(bleurt))

    def test_load_metric(self):
        self.mbr_config.metric = "chrf"
        metric = self.metric_runner._load_metric()
        self.assertIsInstance(metric, evaluate.Metric)
        self.assertEqual(metric.name, "chr_f")
        self.mbr_config.metric = evaluate.load("chrf")
        metric = self.metric_runner._load_metric()
        self.assertIsInstance(metric, evaluate.Metric)
        self.assertEqual(metric.name, "chr_f")

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_metric_config_name(self):
        self.mbr_config.metric = "comet"
        self.mbr_config.metric_config_name = "eamt22-cometinho-da"
        self.mbr_config.metric_output_field = "mean_score"
        metric = self.metric_runner._load_metric()
        self.assertIsInstance(metric, evaluate.Metric)
        self.assertEqual(metric.name, "comet")
        # Test custom metric_config_name
        self.assertEqual(metric.scorer.encoder.__class__.__name__, "MiniLMEncoder")

    def test_compute_metric__chrf(self):
        metric_output = self.metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertTrue(torch.is_floating_point(metric_output.scores_per_reference))
        torch.testing.assert_close(metric_output.scores_per_reference.mean(dim=-1), metric_output.scores)
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        self.assertEqual(metric_output.scores_per_reference.shape, (2, 3, 2))  # batch_size x num_samples x num_references
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        torch.testing.assert_close(metric_output.scores_per_reference[0, 0, 0], metric_output.scores_per_reference[0, 1, 0])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_compute_metric__comet(self):
        self.mbr_config.metric = evaluate.load("comet", "eamt22-cometinho-da")
        self.mbr_config.metric.scorer.eval()
        self.mbr_config.metric_output_field = "mean_score"
        self.metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        self.assertEqual(self.metric_runner.metric.name, "comet")
        metric_output = self.metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertTrue(torch.is_floating_point(metric_output.scores_per_reference))
        torch.testing.assert_close(metric_output.scores_per_reference.mean(dim=-1), metric_output.scores)
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        self.assertEqual(metric_output.scores_per_reference.shape, (2, 3, 2))  # batch_size x num_samples x num_references
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        torch.testing.assert_close(metric_output.scores_per_reference[0, 0, 0], metric_output.scores_per_reference[0, 1, 0])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_compute_metric__bleurt(self):
        self.mbr_config.metric = evaluate.load("bleurt")
        self.mbr_config.metric_output_field = "scores"
        self.metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        self.assertEqual(self.metric_runner.metric.name, "bleurt")
        metric_output = self.metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertTrue(torch.is_floating_point(metric_output.scores_per_reference))
        torch.testing.assert_close(metric_output.scores_per_reference.mean(dim=-1), metric_output.scores)
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        self.assertEqual(metric_output.scores_per_reference.shape, (2, 3, 2))  # batch_size x num_samples x num_references
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        torch.testing.assert_close(metric_output.scores_per_reference[0, 0, 0], metric_output.scores_per_reference[0, 1, 0])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_comet_metric_runner(self):
        from mbr.metrics.comet import CometMetricRunner
        self.mbr_config.metric = evaluate.load("comet", "eamt22-cometinho-da")
        self.mbr_config.metric.scorer.eval()
        self.mbr_config.metric_output_field = "mean_score"
        base_metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        self.assertEqual(base_metric_runner.metric.name, "comet")
        self.assertFalse(base_metric_runner.metric.scorer.training)
        comet_metric_runner = CometMetricRunner(self.mbr_config, self.tokenizer)
        self.assertFalse(comet_metric_runner.metric.scorer.training)
        # Output should be the same as the base MetricRunner
        base_metric_scores = base_metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        metric_scores = comet_metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        torch.testing.assert_close(base_metric_scores, metric_scores)

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_comet_metric_runner__cache(self):
        """Output should be identical irrespective of cache size"""
        from mbr.metrics.comet import CometMetricRunner
        self.mbr_config.metric = evaluate.load("comet", "eamt22-cometinho-da")
        self.mbr_config.metric_output_field = "mean_score"
        base_metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        base_metric_scores = base_metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertEqual(base_metric_runner.metric.name, "comet")
        for cache_size in [1, 4, 8]:
            self.mbr_config.metric_cache_size = cache_size
            comet_metric_runner = CometMetricRunner(self.mbr_config, self.tokenizer)
            metric_scores = comet_metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
            torch.testing.assert_close(base_metric_scores, metric_scores)

    @unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
    def test_comet_metric_runner__aggregate(self):
        from mbr.metrics.comet import AggregateCometMetricRunner
        self.mbr_config.metric = evaluate.load("comet", "eamt22-cometinho-da")
        self.mbr_config.metric.scorer.eval()
        self.mbr_config.metric_output_field = "mean_score"
        base_metric_runner = MetricRunner(self.mbr_config, self.tokenizer)
        self.assertEqual(base_metric_runner.metric.name, "comet")
        self.assertFalse(base_metric_runner.metric.scorer.training)
        comet_metric_runner = AggregateCometMetricRunner(self.mbr_config, self.tokenizer)
        self.assertFalse(comet_metric_runner.metric.scorer.training)
        metric_output = comet_metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertIsNone(metric_output.scores_per_reference)
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])

    def test_fastchrf_metric_runner__aggregate(self):
        from mbr.metrics.fastchrf import FastChrfMetricRunner
        metric_runner = FastChrfMetricRunner(self.mbr_config, self.tokenizer, compute_pairwise_average=False)
        metric_output = metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertIsNone(metric_output.scores_per_reference)
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])

    def test_fastchrf_metric_runner__pairwise(self):
        from mbr.metrics.fastchrf import FastChrfMetricRunner
        metric_runner = FastChrfMetricRunner(self.mbr_config, self.tokenizer, compute_pairwise_average=True)
        metric_output = metric_runner(self.input_ids, self.sample_ids, self.reference_ids)
        self.assertTrue(torch.is_floating_point(metric_output.scores))
        self.assertTrue(torch.is_floating_point(metric_output.scores_per_reference))
        self.assertEqual(metric_output.scores.shape, (2, 3))  # batch_size x num_samples
        self.assertEqual(metric_output.scores_per_reference.shape, (2, 3, 2))  # batch_size x num_samples x num_references
        # Duplicate samples should have the same scores
        torch.testing.assert_close(metric_output.scores[0, 0], metric_output.scores[0, 1])
        torch.testing.assert_close(metric_output.scores_per_reference[0, 0, 0], metric_output.scores_per_reference[0, 1, 0])
        # The metric scores should rank as expected, given the test strings in self.samples and self.references
        self.assertGreater(metric_output.scores[0, 0], metric_output.scores[0, 2])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 1])
        self.assertLess(metric_output.scores[1, 0], metric_output.scores[1, 2])
