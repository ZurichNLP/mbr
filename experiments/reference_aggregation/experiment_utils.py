from dataclasses import dataclass
from pathlib import Path
from typing import List

import sacrebleu


SEEDS = [
    553589,
    456178,
    817304,
    6277,
    792418,
    707983,
    249859,
    272618,
    760402,
    472974,
]


@dataclass
class Testset:
    testset: str
    language_pair: str
    source_sentences: List[str]
    references: List[str]

    @property
    def src_lang(self):
        return self.language_pair.split("-")[0]

    @property
    def tgt_lang(self):
        return self.language_pair.split("-")[1]

    @classmethod
    def from_wmt(cls, wmt: str, language_pair: str, limit_segments: int = None):
        assert wmt in {"wmt21", "wmt22"}
        src_path = sacrebleu.get_source_file(wmt, language_pair)
        ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
        source_sequences = Path(src_path).read_text().splitlines()
        references = Path(ref_path).read_text().splitlines()
        assert len(source_sequences) == len(references)
        if limit_segments is not None:
            source_sequences = source_sequences[:limit_segments]
            references = references[:limit_segments]
        return cls(testset=wmt, language_pair=language_pair, source_sentences=source_sequences, references=references, )

    def __str__(self):
        return f"{self.testset}.{self.language_pair}"
