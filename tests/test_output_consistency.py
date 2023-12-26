from unittest import TestCase

import torch.testing
from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed

from mbr import MBR, MBRGenerationConfig, MBROutput


class OutputConsistencyTestCase(TestCase):
    """
    Test that the output of MBR remains the same across different versions of this library.
    """

    def setUp(self):
        self.model = MBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def test_output(self):
        set_seed(42)
        mbr_config = MBRGenerationConfig(
            num_samples=5,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            output_all_samples=True,
            output_reference_sequences=True,
            output_metric_scores=True,
        )
        input_sentences = [
            "Hello, my name is",
            "This is another sentence because",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output: MBROutput = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        torch.testing.assert_close(output.sequences[0],
                                   torch.tensor([15496, 11, 616, 1438, 318, 3977, 11, 290, 314, 716,
                                                 262, 1772, 286, 257, 11648, 1444, 366, 464, 7443, 286]))
        torch.testing.assert_close(output.selected_samples_indices, torch.tensor([1, 1]))
        torch.testing.assert_close(output.references[0].sequences, torch.tensor(
            [[15496, 11, 616, 1438, 318, 449, 13, 41, 13, 53, 13, 447, 103, 290, 356, 423, 257, 1049, 6180, 13],
             [1212, 318, 1194, 6827, 780, 612, 373, 2147, 2642, 351, 340, 13, 447, 237, 198, 1532, 484, 547, 284,
              423]]))
        torch.testing.assert_close(output.metric_scores, torch.tensor(
            [[43.1201, 46.1530, 43.5142, 43.8980, 44.0345],
             [57.1227, 57.2903, 54.9877, 57.1268, 56.8152]]))
