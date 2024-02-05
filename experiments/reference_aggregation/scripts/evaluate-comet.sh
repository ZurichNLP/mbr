#!/bin/bash

model=$1
if [ "$model" != "Unbabel/wmt22-comet-da" ] && [ "$model" != "Unbabel/eamt22-cometinho-da" ] && [ "$model" != "Unbabel/XCOMET-XL" ]; then
    echo "Invalid model. Please choose one of Unbabel/wmt22-comet-da, Unbabel/eamt22-cometinho-da, Unbabel/XCOMET-XL"
    exit 1
fi

cd translations

declare -a language_pairs=("en-de" "de-en" "en-ru" "ru-en")

for lp in "${language_pairs[@]}"
do
    echo $lp

    IFS='-' read -ra ADDR <<< "$lp"
    src=${ADDR[0]}
    tgt=${ADDR[1]}

    echo "baselines"
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t wmt22.${lp}.beam4.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t wmt22.${lp}.epsilon0.02.seed0.${tgt} --only_system --model $1

    echo "chrf"
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.chrf.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.chrf.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf.${tgt} --only_system --model $1

    echo "cometinho"
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.cometinho.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.cometinho.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.cometinho.${tgt} --only_system --model $1

    echo "comet22"
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.comet22.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.comet22.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.comet22.${tgt} --only_system --model $1

    echo "coarse-to-fine"
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.coarse_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} --only_system --model $1
    comet-score -s wmt22.${lp}.src.${src} -r wmt22.${lp}.ref.${tgt} -t mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} --only_system --model $1

    echo
done
