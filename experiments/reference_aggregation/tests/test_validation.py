import logging
from pathlib import Path
from unittest import TestCase

import jsonlines

logging.basicConfig(level=logging.INFO)


class ValidationTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_run_validation_cometinho(self):
        from experiments.reference_aggregation.validation import main
        jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="cometinho", topk=4, num_samples=8,
                          limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(jsonl_path.exists())
        with jsonlines.open(jsonl_path) as f:
            data = list(f)

        n_by_s_lines = [line for line in data if line["method"] == "n_by_s"]
        s_values = [line["s"] for line in n_by_s_lines]
        self.assertEqual([8, 4, 2, 1], s_values)

        aggregate_lines = [line for line in data if line["method"] == "aggregate"]
        s_values = [line["s"] for line in aggregate_lines]
        self.assertEqual([8, 4, 2, 1], s_values)

        for line in data:
            self.assertEqual(4, len(line["rankings"]))
            self.assertEqual(4, len(line["rankings"][0]))
            self.assertEqual(4, len(set(line["rankings"][0])))

        test_translation_paths = [
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.n_by_s.s8.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.n_by_s.s4.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.n_by_s.s2.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.n_by_s.s1.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.aggregate.s8.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.aggregate.s4.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.aggregate.s2.{self.language_pair.split('-')[1]}",
            self.test_dir / "translations" / f"validation.{self.testset}.{self.language_pair}.n8.epsilon0.02.seed0.cometinho.aggregate.s1.{self.language_pair.split('-')[1]}", ]
        for translation_path in test_translation_paths:
            self.assertTrue(translation_path.exists())
            self.assertIn(self.test_dir, translation_path.parents)
            self.assertTrue(translation_path.name.endswith(".de"))
            translations = translation_path.read_text().splitlines()
            self.assertEqual(len(translations), 4)

    def test_run_validation_topk_first_identical(self):
        """
        The top translation should be the same of any k
        """
        from experiments.reference_aggregation.validation import main

        # cometinho
        top1_jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="cometinho", topk=1,
                               num_samples=8, limit_segments=1, out_dir=self.test_dir)
        with jsonlines.open(top1_jsonl_path) as f:
            top1_best_indices = [line["rankings"][0][0] for line in f]
        for k in [1, 2, 4, 8]:
            jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="cometinho", topk=k,
                              num_samples=8, limit_segments=1, out_dir=self.test_dir)
            with jsonlines.open(jsonl_path) as f:
                best_indices = [line["rankings"][0][0] for line in f]
            self.assertEqual(top1_best_indices, best_indices)

        # chrf
        top1_jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="chrf", topk=1, num_samples=8,
                               limit_segments=1, out_dir=self.test_dir)
        with jsonlines.open(top1_jsonl_path) as f:
            top1_best_indices = [line["rankings"][0][0] for line in f]
        for k in [1, 2, 4, 8]:
            jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="chrf", topk=k, num_samples=8,
                              limit_segments=1, out_dir=self.test_dir)
            with jsonlines.open(jsonl_path) as f:
                best_indices = [line["rankings"][0][0] for line in f]
            self.assertEqual(top1_best_indices, best_indices)

    def test_plot_accuracy(self):
        # Run validation.py
        from experiments.reference_aggregation.validation import main as validation
        jsonl_path = validation(self.testset, self.language_pair, seed_no=0, utility_name="cometinho", topk=8,
                                num_samples=8, limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(jsonl_path.exists())

        # Top-8
        from experiments.reference_aggregation.plot_accuracy import main as plot_accuracy
        series_n_by_s_top8 = plot_accuracy(self.testset, self.language_pair, seed_no=0, fine_utility_name="cometinho",
                                           topk=8, method="n_by_s", num_samples=8, accuracy_topk=8, limit_segments=4,
                                           out_dir=self.test_dir)
        series_aggregate_top8 = plot_accuracy(self.testset, self.language_pair, seed_no=0,
                                              fine_utility_name="cometinho", topk=8, method="aggregate", num_samples=8,
                                              accuracy_topk=8, limit_segments=4, out_dir=self.test_dir)

        # Top-1
        series_n_by_s_top1 = plot_accuracy(self.testset, self.language_pair, seed_no=0, fine_utility_name="cometinho",
                                           topk=8, accuracy_topk=1, method="n_by_s", num_samples=8, limit_segments=4,
                                           out_dir=self.test_dir)
        series_aggregate_top1 = plot_accuracy(self.testset, self.language_pair, seed_no=0,
                                              fine_utility_name="cometinho", topk=8, accuracy_topk=1,
                                              method="aggregate", num_samples=8, limit_segments=4,
                                              out_dir=self.test_dir)

        # Assert that top-1 accuracy <= top-8 accuracy
        for (s, accuracy_top8), (_, accuracy_top1) in zip(series_n_by_s_top8, series_n_by_s_top1):
            self.assertLessEqual(accuracy_top1, accuracy_top8)
        for (s, accuracy_top8), (_, accuracy_top1) in zip(series_aggregate_top8, series_aggregate_top1):
            self.assertLessEqual(accuracy_top1, accuracy_top8)

        # Top-4
        series_n_by_s_top4 = plot_accuracy(self.testset, self.language_pair, seed_no=0, fine_utility_name="cometinho",
                                           topk=8, accuracy_topk=4, method="n_by_s", num_samples=8, limit_segments=4,
                                           out_dir=self.test_dir)
        series_aggregate_top4 = plot_accuracy(self.testset, self.language_pair, seed_no=0,
                                              fine_utility_name="cometinho", topk=8, accuracy_topk=4,
                                              method="aggregate", num_samples=8, limit_segments=4,
                                              out_dir=self.test_dir)

        # Assert that top-4 accuracy <= top-8 accuracy
        for (s, accuracy_top8), (_, accuracy_top4) in zip(series_n_by_s_top8, series_n_by_s_top4):
            self.assertLessEqual(accuracy_top4, accuracy_top8)
        for (s, accuracy_top8), (_, accuracy_top4) in zip(series_aggregate_top8, series_aggregate_top4):
            self.assertLessEqual(accuracy_top4, accuracy_top8)

        # Assert that top-1 accuracy <= top-4 accuracy
        for (s, accuracy_top4), (_, accuracy_top1) in zip(series_n_by_s_top4, series_n_by_s_top1):
            self.assertLessEqual(accuracy_top1, accuracy_top4)
        for (s, accuracy_top4), (_, accuracy_top1) in zip(series_aggregate_top4, series_aggregate_top1):
            self.assertLessEqual(accuracy_top1, accuracy_top4)
