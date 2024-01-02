import os
import unittest
from unittest import TestCase

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, M2M100ForConditionalGeneration, GenerationConfig
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from mbr import MBR, MBRConfig, MBROutput, MetricOutput
from mbr.metrics import load_metric_runner


class DecoderOnlyTestCase(TestCase):

    def setUp(self):
        self.model = MBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def test_generate(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))

    def test_model_output(self):
        mbr_config = MBRConfig(
            num_samples=5,
            return_dict_in_generate=True,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(1, output.sequences.shape[0])
        str_output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertTrue(str_output[0].startswith("Hello, my name is"))
        self.assertIsNone(output.all_samples)
        self.assertIsNone(output.selected_samples_indices)
        self.assertIsNone(output.references)
        self.assertIsNone(output.metric_scores)

    def test_model_output_extended(self):
        mbr_config = MBRConfig(
            metric="pairwise_chrf",
            num_samples=5,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
            output_hidden_states=True,
            output_all_samples=True,
            output_metric_scores=True,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(1, output.sequences.shape[0])
        self.assertIsNotNone(output.all_samples)
        self.assertEqual(5, len(output.all_samples))
        self.assertIsInstance(output.all_samples[0], SampleDecoderOnlyOutput)
        self.assertEqual(1, output.all_samples[0].sequences.shape[0])
        self.assertIsNotNone(output.selected_samples_indices)
        self.assertEqual(1, len(output.selected_samples_indices))
        self.assertIsNotNone(output.references)
        self.assertEqual(5, len(output.references))
        self.assertIsInstance(output.references[0], SampleDecoderOnlyOutput)
        self.assertIsNotNone(output.metric_scores)
        self.assertIsInstance(output.metric_scores, MetricOutput)
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores))
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores_per_reference))
        self.assertEqual([1, 5], list(output.metric_scores.scores.shape))
        self.assertEqual([1, 5, 5], list(output.metric_scores.scores_per_reference.shape))

        # Test the model output for a selected sample
        sample = output.all_samples[output.selected_samples_indices[0]]
        if output.sequences[0].shape[0] <= sample.sequences[0].shape[0]:
            torch.testing.assert_close(output.sequences[0], sample.sequences[0][:output.sequences[0].shape[0]])
        else:
            torch.testing.assert_close(output.sequences[0][:sample.sequences[0].shape[0]], sample.sequences[0])
        self.assertIsNotNone(sample.scores)
        self.assertEqual(1, len(sample.scores[0]))
        self.assertIsNotNone(sample.attentions)
        self.assertEqual(1, len(sample.attentions[0][0]))
        self.assertIsNotNone(sample.hidden_states)
        self.assertEqual(1, len(sample.hidden_states[0][0]))

    def test_metric_runner(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        metric_runner = load_metric_runner(mbr_config, self.tokenizer)
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            metric_runner=metric_runner,
            do_sample=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))

    def test_generation_config(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        generation_config = GenerationConfig.from_pretrained("distilgpt2",
            do_sample=True,
            num_beams=1,
            top_p=0.9,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            generation_config=generation_config,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))

    def test_references_config(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        generation_config = GenerationConfig.from_pretrained("distilgpt2",
            do_sample=True,
            num_beams=1,
            top_p=0.9,
        )
        references_config = GenerationConfig.from_pretrained("distilgpt2",
            do_sample=True,
            num_beams=1,
            epsilon_cutoff=3e-4,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            generation_config=generation_config,
            references_config=references_config,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
class EncoderDecoderTestCase(TestCase):

    def setUp(self):
        self.model = MBR(M2M100ForConditionalGeneration).from_pretrained("alirezamsh/small100").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
        self.tokenizer.tgt_lang = "fr"

    def test_generate(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        input_sentences = [
            "Could you translate this for me, please?",
            "This is another sentence.",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt", padding=True)
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            num_beams=1,
        )
        translations = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(2, len(translations))

    def test_model_output(self):
        mbr_config = MBRConfig(
            num_samples=5,
            return_dict_in_generate=True,
        )
        input_sentences = [
            "Could you translate this for me, please?",
            "This is another sentence.",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt", padding=True)
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            num_beams=1,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(2, output.sequences.shape[0])
        self.assertIsNone(output.all_samples)
        self.assertIsNone(output.selected_samples_indices)
        self.assertIsNone(output.references)
        self.assertIsNone(output.metric_scores)

    def test_model_output_extended(self):
        mbr_config = MBRConfig(
            metric="pairwise_chrf",
            num_samples=5,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
            output_hidden_states=True,
            output_all_samples=True,
            output_metric_scores=True,
        )
        input_sentences = [
            "Could you translate this for me, please?",
            "This is another sentence.",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt", padding=True)
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            num_beams=1,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(2, output.sequences.shape[0])
        self.assertIsNotNone(output.all_samples)
        self.assertEqual(5, len(output.all_samples))
        self.assertIsInstance(output.all_samples[0], SampleEncoderDecoderOutput)
        self.assertEqual(2, output.all_samples[0].sequences.shape[0])
        self.assertIsNotNone(output.selected_samples_indices)
        self.assertEqual(2, len(output.selected_samples_indices))
        self.assertIsNotNone(output.references)
        self.assertEqual(5, len(output.references))
        self.assertIsInstance(output.references[0], SampleEncoderDecoderOutput)
        self.assertIsNotNone(output.metric_scores)
        self.assertIsInstance(output.metric_scores, MetricOutput)
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores))
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores_per_reference))
        self.assertEqual([2, 5], list(output.metric_scores.scores.shape))
        self.assertEqual([2, 5, 5], list(output.metric_scores.scores_per_reference.shape))

        # Test the model output for a selected sample (batch index 0)
        sample = output.all_samples[output.selected_samples_indices[0]]
        if output.sequences[0].shape[0] <= sample.sequences[0].shape[0]:
            torch.testing.assert_close(output.sequences[0], sample.sequences[0][:output.sequences[0].shape[0]])
        else:
            torch.testing.assert_close(output.sequences[0][:sample.sequences[0].shape[0]], sample.sequences[0])
        self.assertIsNotNone(sample.scores)
        self.assertEqual(2, len(sample.scores[0]))
        self.assertIsNotNone(sample.encoder_attentions)
        self.assertEqual(2, len(sample.encoder_attentions[0]))
        self.assertIsNotNone(sample.encoder_hidden_states)
        self.assertEqual(2, len(sample.encoder_hidden_states[0]))
        self.assertIsNotNone(sample.decoder_attentions)
        self.assertEqual(2, len(sample.decoder_attentions[0][0]))
        self.assertIsNotNone(sample.cross_attentions)
        self.assertEqual(2, len(sample.cross_attentions[0][0]))
        self.assertIsNotNone(sample.decoder_hidden_states)
        self.assertEqual(2, len(sample.decoder_hidden_states[0][0]))

        # Test the model output for a selected sample (batch index 1)
        sample = output.all_samples[output.selected_samples_indices[1]]
        if output.sequences[1].shape[0] <= sample.sequences[1].shape[0]:
            torch.testing.assert_close(output.sequences[1], sample.sequences[1][:output.sequences[1].shape[0]])
        else:
            torch.testing.assert_close(output.sequences[1][:sample.sequences[1].shape[0]], sample.sequences[1])
        self.assertIsNotNone(sample.scores)
        self.assertEqual(2, len(sample.scores[0]))
