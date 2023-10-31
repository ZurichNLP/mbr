This directory uses the [**mbr**](https://github.com/ZurichNLP/mbr) package to reproduce an experiment from the paper [Epsilon Sampling Rocks: Investigating Sampling Strategies for Minimum Bayes Risk Decoding for Machine Translation](https://arxiv.org/abs/2305.09860) (Freitag et al., 2023).

## Setup
* Task: Machine translation
* Translation directions: en–de, de–en
* MBR metric: Neural metric (paper: BLEURT, this reproduction: Cometinho)
* Number of samples: 1024
* Various sampling approaches
* Samples and references are the same
* Test set: newstest2021
* Evaluation metric: COMET ([Rei et al., 2020](https://aclanthology.org/2020.emnlp-main.213/))
* Baseline: beam search with beam size 4

## Differences to the paper
* The paper used custom models trained without label smoothing, this reproduction uses an open-source model ([Ng et al., WMT 2019](https://aclanthology.org/W19-5333/)).
* The paper used BLEURT ([Sellam et al., 2020](https://aclanthology.org/2020.acl-main.704/)) as a metric, this reproduction uses Cometinho ([Rei et al., 2022](https://aclanthology.org/2022.eamt-1.9/)).

## Results

Comparison between ancestral sampling and epsilon sampling:

|                              Paper                              | Reproduction |
|:---------------------------------------------------------------:|:---:|
| ![Main Comparison EN–DE (original)](results/figures/Main%20Comparison%20EN–DE%20(original).png) | ![Main Comparison EN–DE (reproduction)](results/figures/Main%20Comparison%20EN–DE%20(reproduction).png) |
| ![Main Comparison DE–EN (original)](results/figures/Main%20Comparison%20DE–EN%20(original).png) | ![Main Comparison DE–EN (reproduction)](results/figures/Main%20Comparison%20DE–EN%20(reproduction).png) |

Comparison between beam search and various sampling approaches:

|                              Paper                              | Reproduction |
|:---------------------------------------------------------------:|:---:|
| ![All Results EN–DE (original)](results/figures/All%20Results%20EN–DE%20(original).png) | ![All Results EN–DE (reproduction)](results/figures/All%20Results%20EN–DE%20(reproduction).png) |
| ![All Results DE–EN (original)](results/figures/All%20Results%20DE–EN%20(original).png) | ![All Results DE–EN (reproduction)](results/figures/All%20Results%20DE–EN%20(reproduction).png) |

Although the models used in this reproduction seem to be less suitable for sampling, especially at higher temperatures, the main comparison between ancestral sampling and epsilon sampling has the same trend as in the paper.
