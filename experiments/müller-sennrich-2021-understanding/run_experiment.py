import sys
from pathlib import Path

import jsonlines
import sacrebleu
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MarianMTModel, AutoTokenizer, pipeline, set_seed
from transformers.pipelines.base import KeyDataset

from mbr import MBR, MBRConfig

set_seed(42)

language_pair = sys.argv[1]

opus_mt_models = {
    "dan-epo": "Helsinki-NLP/opus-mt-da-eo",
    "aze-eng": "Helsinki-NLP/opus-mt-az-en",
    "bel-rus": "Helsinki-NLP/opus-mt-tc-big-zle-zle",
    "deu-fra": "Helsinki-NLP/opus-mt-de-fr",
}

language_codes = {
    "dan": "da",
    "epo": "eo",
    "aze": "az",
    "eng": "en",
    "bel": "be",
    "rus": "ru",
    "deu": "de",
    "fra": "fr",
}

results_file = jsonlines.open(Path(__file__).parent / f"results_{language_pair}.jsonl", "w")

model_name = opus_mt_models[language_pair]
model = MBR(MarianMTModel).from_pretrained(model_name)
model = model.half()
tokenizer = AutoTokenizer.from_pretrained(model_name)
src_code = language_codes[language_pair.split("-")[0]]
tgt_code = language_codes[language_pair.split("-")[1]]
mt_pipeline = pipeline(
    "translation_" + src_code + "_to_" + tgt_code,
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

dataset = load_dataset("Helsinki-NLP/tatoeba_mt", language_pair=language_pair)
references = dataset["test"]["targetString"]

# MBR
mbr_config = MBRConfig()
mbr_config.metric = "chrf"
mbr_config.metric_output_field = "score"
batch_size = 64 if "-big-" in model_name else 256

for num_samples in range(5, 101, 5):
    mbr_config.num_samples = num_samples
    mbr_config.num_references = num_samples

    outputs = mt_pipeline(
        KeyDataset(dataset["test"], "sourceString"),
        mbr_config=mbr_config,
        tokenizer=tokenizer,
        do_sample=True,
        num_beams=1,
        batch_size=batch_size,
    )
    translations = []
    for batch in tqdm(outputs, total=len(dataset["test"]) // batch_size):
        translations += [translation["translation_text"] for translation in batch]
    chrf_score = sacrebleu.corpus_chrf(translations, [references])
    results_file.write({
        "language_pair": language_pair,
        "method": "mbr",
        "num_samples": num_samples,
        "chrf": chrf_score.score,
        "translations": translations,
    })

# Beam search
model = MarianMTModel.from_pretrained(model_name).to(mt_pipeline.device)
mt_pipeline.model = model

outputs = mt_pipeline(
    KeyDataset(dataset["test"], "sourceString"),
    num_beams=5,
    batch_size=32,
)
translations = []
for batch in tqdm(outputs):
    translations += [translation["translation_text"] for translation in batch]
chrf_score = sacrebleu.corpus_chrf(translations, [references])
results_file.write({
    "language_pair": language_pair,
    "method": "beam_search",
    "chrf": chrf_score.score,
    "translations": translations,
})

results_file.close()
