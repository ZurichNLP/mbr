"""
Adapted from https://github.com/huggingface/transformers/blob/main/tests/generation/test_configuration_utils.py
"""

import copy
import tempfile
from unittest import TestCase

from transformers import AutoConfig

from mbr import MBRGenerationConfig


class GenerationConfigTest(TestCase):

    def test_save_load_config(self):
        config = MBRGenerationConfig(
            num_samples=100,
            num_references=50,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = MBRGenerationConfig.from_pretrained(tmp_dir)

        # Checks parameters that were specified
        self.assertEqual(loaded_config.num_samples, 100)
        self.assertEqual(loaded_config.num_references, 50)

        # Checks parameters that were not specified (defaults)
        self.assertEqual(loaded_config.metric , "chrf")

    def test_from_model_config(self):
        model_config = AutoConfig.from_pretrained("distilgpt2")
        generation_config_from_model = MBRGenerationConfig.from_model_config(model_config)
        self.assertIsInstance(generation_config_from_model, MBRGenerationConfig)

    def test_update(self):
        generation_config = MBRGenerationConfig()
        update_kwargs = {
            "num_samples": 200,
            "foo": "bar",
        }
        update_kwargs_copy = copy.deepcopy(update_kwargs)
        unused_kwargs = generation_config.update(**update_kwargs)

        # update_kwargs was not modified (no side effects)
        self.assertEqual(update_kwargs, update_kwargs_copy)

        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(generation_config.num_samples, 200)

        # `.update()` returns a dictionary of unused kwargs
        self.assertEqual(unused_kwargs, {"foo": "bar"})

    def test_initialize_new_kwargs(self):
        generation_config = MBRGenerationConfig()
        generation_config.foo = "bar"

        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)

            new_config = MBRGenerationConfig.from_pretrained(tmp_dir)
        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(new_config.foo, "bar")

        generation_config = MBRGenerationConfig.from_model_config(new_config)
        assert not hasattr(generation_config, "foo")  # no new kwargs should be initialized if from config

    def test_kwarg_init(self):
        """Tests that we can overwrite attributes at `from_pretrained` time."""
        default_config = MBRGenerationConfig()
        self.assertEqual(default_config.num_samples, 10)
        self.assertEqual(default_config.num_references, 10)
        config = MBRGenerationConfig(
            num_samples=100,
            num_references=50,
        )
        self.assertEqual(config.num_samples, 100)
        self.assertEqual(config.num_references, 50)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = MBRGenerationConfig.from_pretrained(tmp_dir, num_samples=200)

        self.assertEqual(loaded_config.num_samples, 200)
        self.assertEqual(loaded_config.num_references, 50)
        self.assertEqual(loaded_config.metric, "chrf")  # default value
