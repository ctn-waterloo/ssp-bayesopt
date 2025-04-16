#!/bin/bash


#for taskID in "pest" "rna_inverse_fold"
#do
#  for seed in {0..9}
#  do
#    python  experiments/run_agent_mcbo.py \
#            --task-id $taskID \
#            --ssp-dim 201 \
#            --num-samples 200 \
#            --num-init-samples 20 \
#            --beta-ucb 14.5. \
#            --alpha-decay 1.0 \
#            --length-scale 4.0 \
#            --data-dir "data/mcbo" \
#            --seed $seed
#  done
#done
#"xgboost_opt"
for taskID in  "svm_opt"
do
  for seed in {0..9}
  do
    python  experiments/run_agent_mcbo.py \
            --task-id $taskID \
            --ssp-dim 901 \
            --num-samples 200 \
            --num-init-samples 20 \
            --beta-ucb 100. \
            --alpha-decay 0.99 \
            --length-scale 4.0 \
            --data-dir "data/mcbo" \
            --seed $seed
  done
done

#conda deactivate
#conda activate testenv

#for seed in {0..9}
#do
#  python  experiments/run_agent_graphs.py \
#          --nas-data-dir "./experiments/nas_data" \
#          --ssp-dim 301 \
#          --num-samples 200 \
#          --num-init-samples 8 \
#          --beta-ucb 14.5 \
#          --data-dir "./data/nasbench" \
#          --seed $seed
#done
