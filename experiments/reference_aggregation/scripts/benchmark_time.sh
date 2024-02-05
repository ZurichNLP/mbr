#/bin/bash

num_segments_per_lp=32

for lp in en-de de-en en-ru ru-en; do
  echo $lp

  # MBR with ChrF metric – standard MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method pairwise --testset wmt22 --language-pair $lp --seed 0 --utility chrf --limit-segments $num_segments_per_lp --log-time
  # MBR with ChrF metric – reference aggregation
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate --testset wmt22 --language-pair $lp --seed 0 --utility chrf --limit-segments $num_segments_per_lp --log-time
  # MBR with ChrF metric – aggregate-to-fine MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate_to_fine --topk 20 --testset wmt22 --language-pair $lp --seed 0 --utility chrf --limit-segments $num_segments_per_lp --log-time

  # MBR with Comethinho metric – standard MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method pairwise --testset wmt22 --language-pair $lp --seed 0 --utility cometinho --limit-segments $num_segments_per_lp --log-time
  # MBR with Cometinho metric – reference aggregation
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate --testset wmt22 --language-pair $lp --seed 0 --utility cometinho --limit-segments $num_segments_per_lp --log-time
  # MBR with Cometinho metric – aggregate-to-fine MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate_to_fine --topk 20 --testset wmt22 --language-pair $lp --seed 0 --utility cometinho --limit-segments $num_segments_per_lp --log-time

  # MBR with COMET-22 metric – standard MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method pairwise --testset wmt22 --language-pair $lp --seed 0 --utility comet22 --limit-segments $num_segments_per_lp --log-time
  # MBR with COMET-22 metric – reference aggregation
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate --testset wmt22 --language-pair $lp --seed 0 --utility comet22 --limit-segments $num_segments_per_lp --log-time
  # MBR with COMET-22 metric – aggregate-to-fine MBR
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate_to_fine --topk 20 --testset wmt22 --language-pair $lp --seed 0 --utility comet22 --limit-segments $num_segments_per_lp --log-time

  # Coarse-to-fine MBR: ChrF to COMET-22
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method coarse_to_fine --topk 20 --testset wmt22 --language-pair $lp --seed 0 --coarse-utility chrf --utility comet22 --limit-segments $num_segments_per_lp --log-time
  # Aggregate-to-fine MBR: Aggregate ChrF to COMET-22
  taskset --cpu-list 0-63 python -m experiments.reference_aggregation.run_mbr --method aggregate_to_fine --topk 20 --testset wmt22 --language-pair $lp --seed 0 --coarse-utility chrf --utility comet22 --limit-segments $num_segments_per_lp --log-time

  echo

done
