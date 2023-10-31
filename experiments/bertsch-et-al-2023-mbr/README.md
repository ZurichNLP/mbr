This directory uses the [**mbr**](https://github.com/ZurichNLP/mbr) package to reproduce an experiment from the paper [It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk](https://arxiv.org/abs/2310.01387) (Bertsch et al., 2023).

## Setup
* Task: Summarization
* Language: English
* Model: facebook/bart-large-cnn ([Lewis et al., 2020](https://aclanthology.org/2020.acl-main.703/))
* MBR metric: ROUGE-1 ([Lin, 2004](https://aclanthology.org/W04-1013/))
* Number of samples: 30
* Number of references: 30
* Sampling approach: sampling with temperature 0.5
* Reference sampling approach: sampling with temperature 1.0
* Test set: CNN/DailyMail ([Nallapati et al., 2016](https://aclanthology.org/K16-1028/))
* Evaluation metric: ROUGE-1
* Baselines: greedy decoding, beam search

## Results

| Method             |  ROUGE-1 (original) | ROUGE-1 (reproduction) |
|:-------------------|--------------------:|-----------------------:|
| Greedy             |               43.98 |                  43.20 |
| Beam search (k=5)  |               43.16 |                  42.56 |
| Beam search (k=10) |               42.62 |                    tba |
| MBR ROUGE-1        |               46.89 |                  46.04 |
