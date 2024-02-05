#!/bin/bash

cd translations

declare -a language_pairs=("en-de" "de-en" "en-ru" "ru-en")

for lp in "${language_pairs[@]}"
do
    echo $lp

    IFS='-' read -ra ADDR <<< "$lp"
    src=${ADDR[0]}
    tgt=${ADDR[1]}

    echo "baselines"
    sacrebleu wmt22.${lp}.ref.${tgt} -i wmt22.${lp}.beam4.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i wmt22.${lp}.epsilon0.02.seed0.${tgt} -m chrf -b

    echo "chrf"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b

    echo "cometinho"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b

    echo "comet22"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b

    echo "coarse-to-fine"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.coarse_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} -m chrf -b
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} -m chrf -b

    echo
done
