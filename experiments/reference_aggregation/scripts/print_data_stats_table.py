from pathlib import Path

import jsonlines

header = "\\begin{tabularx}{\\textwidth}{Xrrr}\n\\toprule\n"
header += "& \\# Segments & \\# Samples per segment & \\# Unique samples per segment \\\\\n\\midrule\n"
footer = "\\bottomrule\n\\end{tabularx}"

samples_dir = Path(__file__).parent.parent / "samples"

body = ""
body += "\\textit{newstest21} & & & \\\\\n"
for lang_pair in ["en-de", "de-en", "en-ru", "ru-en"]:
    path = samples_dir / f"samples.wmt21.{lang_pair}.n1024.epsilon0.02.seed0.jsonl"
    assert path.exists(), f"Path {path} does not exist"
    with jsonlines.open(path) as reader:
        data = list(reader)
    num_segments = len(data)
    num_samples = len(data[0]["samples"])
    avg_num_unique_samples = sum(
        [len(set([sample for sample in segment["samples"]])) for segment in data]) / num_segments
    body += "\\textsc{" + lang_pair.replace('-', '–') + "} & " + str(num_segments) + " & " + str(
        num_samples) + " & " + "{:.1f}".format(avg_num_unique_samples) + " \\\\\n"
body += "\\addlinespace\n"
body += "\\textit{newstest22} & & & \\\\\n"
for lang_pair in ["en-de", "de-en", "en-ru", "ru-en"]:
    path = samples_dir / f"samples.wmt22.{lang_pair}.n1024.epsilon0.02.seed0.jsonl"
    assert path.exists(), f"Path {path} does not exist"
    with jsonlines.open(path) as reader:
        data = list(reader)
    num_segments = len(data)
    num_samples = len(data[0]["samples"])
    avg_num_unique_samples = sum(
        [len(set([sample for sample in segment["samples"]])) for segment in data]) / num_segments
    body += "\\textsc{" + lang_pair.replace('-', '–') + "} & " + str(num_segments) + " & " + str(
        num_samples) + " & " + "{:.1f}".format(avg_num_unique_samples) + " \\\\\n"

print(header + body + footer)
