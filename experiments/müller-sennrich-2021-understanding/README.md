This directory uses the [**mbr**](https://github.com/ZurichNLP/mbr) package to reproduce an experiment from the paper [Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation](https://aclanthology.org/2021.acl-long.22) (Müller & Sennrich, ACL-IJCNLP 2021).

## Setup
* Task: Machine translation
* Translation directions: dan–epo, aze–eng, bel–rus, deu–fra
* MBR metric: ChrF2 ([Popović, 2015](https://aclanthology.org/W15-3049/))
* Number of samples: 5–100
* Sampling approach: ancestral sampling
* Samples and references are the same
* Test set: Tatoeba ([Tiedemann, 2020](https://aclanthology.org/2020.wmt-1.139/))
* Evaluation metric: ChrF2
* Baseline: beam search with beam size 5

## Differences to the paper
* The paper used custom models trained without label smoothing, this reproduction uses open-source models from Opus-MT ([Tiedemann & Thottingal, 2020](https://aclanthology.org/2020.eamt-1.61)).
* The paper reports averages over 2 runs, this reproduction uses a single run.

## Results

|                              Paper                              | Reproduction |
|:---------------------------------------------------------------:|:---:|
| ![AZE–ENG (original)](results/figures/AZE–ENG%20(original).png) | ![AZE–ENG (reproduction)](results/figures/AZE–ENG%20(reproduction).png) |
| ![BEL–RUS (original)](results/figures/BEL–RUS%20(original).png) | ![BEL–RUS (reproduction)](results/figures/BEL–RUS%20(reproduction).png) |
| ![DAN–EPO (original)](results/figures/DAN–EPO%20(original).png) | ![DAN–EPO (reproduction)](results/figures/DAN–EPO%20(reproduction).png) |
| ![DEU–FRA (original)](results/figures/DEU–FRA%20(original).png) | ![DEU–FRA (reproduction)](results/figures/DEU–FRA%20(reproduction).png) |