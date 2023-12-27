from copy import deepcopy
from pathlib import Path

import evaluate
import jsonlines
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import BartForConditionalGeneration, AutoTokenizer, pipeline, GenerationConfig

from mbr import MBR, MBRConfig

results_file = jsonlines.open(Path(__file__).parent / f"results.jsonl", "w")

model_name = "facebook/bart-large-cnn"
model = MBR(BartForConditionalGeneration).from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarization_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_rouge = evaluate.load("rouge")

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# MBR
mbr_config = MBRConfig()
mbr_config.num_samples = 30
mbr_config.num_references = 30
mbr_config.metric = "rouge"
mbr_config.metric_output_field = "rouge1"
# efficiency settings
mbr_config.metric_kwargs = {"rouge_types": ("rouge1",), "use_aggregator": False}

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.temperature = 0.5
generation_config.early_stopping = False
generation_config.length_penalty = 1.0

references_config = GenerationConfig.from_pretrained(model_name)
references_config.do_sample = True
references_config.num_beams = 1
references_config.temperature = 1.0
references_config.early_stopping = False
references_config.length_penalty = 1.0

summaries = []
outputs = summarization_pipeline(
    dataset["article"],
    mbr_config=mbr_config,
    generation_config=generation_config,
    references_config=references_config,
    tokenizer=tokenizer,
    truncation=True,
    progress_bar=True,
    batch_size=32,
)
for output in outputs:
    summaries.append(output["summary_text"])
rouge_score = evaluation_metric_rouge.compute(predictions=summaries, references=dataset["highlights"])
results_file.write({
    "method": "mbr rouge-1",
    "rouge": rouge_score,
    "summaries": summaries,
})

# Baselines
model = BartForConditionalGeneration.from_pretrained(model_name).to(summarization_pipeline.device)
summarization_pipeline.model = model
base_generation_config = GenerationConfig.from_pretrained(model_name)
generation_configs = {}

# greedy
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = False
generation_config.num_beams = 1
generation_config.early_stopping = False
generation_config.length_penalty = 1.0
generation_configs["greedy"] = generation_config

# beam search k=5
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = False
generation_config.num_beams = 5
generation_configs["beam search k=5"] = generation_config

# beam search k=10
generation_config = deepcopy(base_generation_config)
generation_config.do_sample = False
generation_config.num_beams = 10
generation_configs["beam search k=10"] = generation_config

for method, generation_config in generation_configs.items():
    print(method, flush=True)
    summaries = []
    outputs = summarization_pipeline(
        dataset["article"],
        generation_config=generation_config,
        truncation=True,
        batch_size=1,
    )
    for output in tqdm(outputs):
        summaries.append(output["summary_text"])
    rouge_score = evaluation_metric_rouge.compute(predictions=summaries, references=dataset["highlights"])
    results_file.write({
        "method": method,
        "rouge": rouge_score,
        "summaries": summaries,
    })

results_file.close()
