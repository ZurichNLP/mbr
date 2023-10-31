import sys
from copy import deepcopy
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, AutoTokenizer, pipeline, set_seed, GenerationConfig

from mbr import MBR, MBRGenerationConfig
from mbr.metrics.comet import CometMetricRunner

set_seed(42)

language_pair = sys.argv[1]

models = {
    "en-de": "facebook/wmt19-en-de",
    "en-zh": None,
    "de-en": "facebook/wmt19-de-en",
    "zh-en": None,
}

testsets = {
    "en-de": "wmt21/C",
    "en-zh": "wmt21/B",
    "de-en": "wmt21",
    "zh-en": "wmt21",
}

results_file = jsonlines.open(Path(__file__).parent / f"results_{language_pair}3.jsonl", "w")

model_name = models[language_pair]
model = MBR(FSMTForConditionalGeneration).from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
mt_pipeline = pipeline(
    "translation_" + language_pair.split("-")[0] + "_to_" + language_pair.split("-")[1],
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_comet = evaluate.load("comet", "wmt20-comet-da")

src_path = sacrebleu.get_source_file(testsets[language_pair], language_pair)
ref_path = sacrebleu.get_reference_files(testsets[language_pair], language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
assert len(dataset["test"]) == len(references)

# MBR
mbr_config = MBRGenerationConfig()
mbr_config.num_samples = 1024
mbr_config.metric = "comet"
mbr_config.metric_config_name = "eamt22-cometinho-da"
mbr_config.metric_output_field = "mean_score"

metric_runner = CometMetricRunner(
    mbr_config,
    tokenizer,
    device=mt_pipeline.device,
    batch_size_embed=64,
    batch_size_estimate=64,
    progress_bar=True,
)

base_generation_config = GenerationConfig.from_pretrained(model_name)
base_generation_config.do_sample = True
base_generation_config.num_beams = 1
base_generation_config.early_stopping = False
generation_configs = {}

# # MBR – Ancestral (τ=1.0)
# generation_config = deepcopy(base_generation_config)
# generation_config.temperature = 1.0
# generation_configs["mbr ancestral (τ=1.0)"] = generation_config
#
# # MBR – Top-k (k=10, τ=1.0)
# generation_config = deepcopy(base_generation_config)
# generation_config.top_k = 10
# generation_config.temperature = 1.0
# generation_configs["mbr top-k (k=10, τ=1.0)"] = generation_config

# # MBR – Top-k (k=50, τ=1.0)
# generation_config = deepcopy(base_generation_config)
# generation_config.top_k = 50
# generation_config.temperature = 1.0
# generation_configs["mbr top-k (k=50, τ=1.0)"] = generation_config
#
# # MBR – Nucleus (p=0.9, τ=1.5)
# generation_config = deepcopy(base_generation_config)
# generation_config.top_p = 0.9
# generation_config.temperature = 1.5
# generation_configs["mbr nucleus (p=0.9, τ=1.5)"] = generation_config
#
# MBR – Epsilon (ε=0.02, τ=1.0)
generation_config = deepcopy(base_generation_config)
generation_config.epsilon_cutoff = 0.02
generation_config.temperature = 1.0
generation_configs["mbr epsilon (ε=0.02, τ=1.0)"] = generation_config

# MBR – Epsilon (ε=0.02, τ=2.0)
generation_config = deepcopy(base_generation_config)
generation_config.epsilon_cutoff = 0.02
generation_config.temperature = 2.0
generation_configs["mbr epsilon (ε=0.02, τ=2.0)"] = generation_config

for method, generation_config in generation_configs.items():
    outputs = mt_pipeline(
        dataset["test"]["text"],
        mbr_config=mbr_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        metric_runner=metric_runner,
        batch_size=32,
        progress_bar=True
    )
    translations = []
    for batch in tqdm(outputs):
        if isinstance(batch, dict):
            batch = [batch]
        translations += [translation["translation_text"] for translation in batch]
    comet_score = evaluation_metric_comet.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
    )
    results_file.write({
        "language_pair": language_pair,
        "method": method,
        "comet20": comet_score["mean_score"],
        "translations": translations,
    })

# Beam search
model = FSMTForConditionalGeneration.from_pretrained(model_name).half().to(mt_pipeline.device)
mt_pipeline.model = model
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.num_beams = 4

outputs = mt_pipeline(
    dataset["test"]["text"],
    generation_config=generation_config,
    batch_size=32,
)
translations = []
for batch in tqdm(outputs):
    if isinstance(batch, dict):
        batch = [batch]
    translations += [translation["translation_text"] for translation in batch]
comet_score = evaluation_metric_comet.compute(
    predictions=translations,
    references=references,
    sources=dataset["test"]["text"],
)
results_file.write({
    "language_pair": language_pair,
    "method": "beam search",
    "comet20": comet_score["mean_score"],
    "translations": translations,
})

results_file.close()
