Comparison of [fastChrF](https://github.com/jvamvas/fastChrF) to standard sentence-level ChrF ([Popović, 2015](https://aclanthology.org/W15-3049/)) as a metric for MBR.

## Setup
* Task: Machine translation
* Translation directions: en–de, de–en, en–ru, ru–en
* Model: [facebook/wmt19-*](https://huggingface.co/facebook/wmt19-en-de) ([Ng et al., 2019](https://aclanthology.org/W19-5333/)).
* MBR metrics: `fastchrf.pairwise_chrf` (a fast implementation of standard ChrF) and `fastchrf.aggregate_chrf` (a streamlined ChrF variant for MBR)
* Number of samples: 256
* Sampling approach: epsilon sampling with ε=0.02
* Samples and references are the same
* Test set: newstest2019
* Evaluation metrics: chrF ([sacreBLEU](https://github.com/mjpost/sacrebleu)) and COMET-22 ([Rei et al., 2022](https://aclanthology.org/2022.wmt-1.52/))
* Baseline: beam search with beam size 4

## Results
| Language Pair | Method                               |     ChrF |     COMET | duration (s) |
|---------------|--------------------------------------|---------:|----------:|-------------:|
| en-de         | MBR with `fastchrf.pairwise_chrf`      |     67.7 |     0.867 |         7798 |
| en-de         | MBR with `fastchrf.aggregate_chrf`     |     67.7 |     0.867 |         7480 |
| en-de         | Beam search                          |     67.7 |     0.868 |           62 |
| de-en         | MBR with `fastchrf.pairwise_chrf`      |     65.4 |     0.851 |         6894 |
| de-en         | MBR with `fastchrf.aggregate_chrf`     |     65.6 |     0.850 |         6849 |
| de-en         | Beam search                          |     65.1 |     0.851 |           53 |
| en-ru         | MBR with `fastchrf.pairwise_chrf`      |     57.5 |     0.862 |         7802 |
| en-ru         | MBR with `fastchrf.aggregate_chrf`     |     57.5 |     0.862 |         7465 |
| en-ru         | Beam search                          |     56.9 |     0.863 |           64 |
| ru-en         | MBR with `fastchrf.pairwise_chrf`      |     64.2 |     0.847 |         7541 |
| ru-en         | MBR with `fastchrf.aggregate_chrf`     |     64.3 |     0.848 |         6689 |
| ru-en         | Beam search                          |     63.5 |     0.847 |           61 |
| **Average**   | **MBR with `fastchrf.pairwise_chrf`**  | **63.7** | **0.857** |     **7509** |
| **Average**   | **MBR with `fastchrf.aggregate_chrf`** | **63.7** | **0.857** |     **7121** |
| **Average**   | **Beam search**                      | **63.3** | **0.857** |       **60** |