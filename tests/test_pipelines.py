import os
import unittest
from unittest import TestCase

from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel, M2M100ForConditionalGeneration, set_seed

from mbr import MBRConfig
from mbr import MBR


class TextGenerationTestCase(TestCase):

    def setUp(self):
        set_seed(42)
        self.model = MBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def test_pipeline(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        output = self.pipeline(
            "Hello,",
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(1, len(output))
        self.assertIn("generated_text", output[0])


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
class TranslationTestCase(TestCase):

    def setUp(self):
        set_seed(42)
        self.model = MBR(M2M100ForConditionalGeneration).from_pretrained("alirezamsh/small100").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
        self.pipeline = pipeline("translation_en_to_fr", model=self.model, tokenizer=self.tokenizer)
        self.tokenizer.tgt_lang = "fr"

    def test_pipeline(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        output = self.pipeline(
            "Could you translate this for me, please?",
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            num_beams=1,
        )
        self.assertEqual(1, len(output))
        self.assertIn("translation_text", output[0])
