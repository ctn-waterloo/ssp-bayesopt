#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=0-6:00
#SBATCH --array=0-13

module load python/3.10 scipy-stack
source ~/projects/def-celiasmi/ns2dumon/bosspenv/bin/activate


agt_index=$(( SLURM_ARRAY_TASK_ID / 7 ))
dim_index=$(( SLURM_ARRAY_TASK_ID % 7 ))

beta=1.0
vardims=(1 2 3 4 5 6 7)
dim="${vardims[dim_index]}"

func="styblinski-tang$dim"
nscales=3

for sspdim in 100 250 500
	do
	  nrotates=$(python -c "import numpy as np; print(int(np.round(($sspdim - 1) / (2 * $nscales * ($dim + 1)))))")
    dest_dir="data/sspdim$sspdim_vardim$dim$"
    if [ "$agt_index" -eq 0 ]; then
        echo "Running ssp-rand with target sspdim of $sspdim"
        python run_agent.py --agent "ssp-rand" --num-trials 10 --func $func --data-dir $dest_dir --len-scale -1 --n-scales $nscales --n-rotates $nrotates --gamma 0.0 --beta-ucb $beta
    elif [ "$agt_index" -eq 1 ]; then
        echo "Running ssp-hex with target sspdim of $sspdim"
        python run_agent.py --agent "ssp-hex" --num-trials 10 --func $func --data-dir $dest_dir --len-scale -1 --n-scales $nscales --n-rotates $nrotates --gamma 0.0 --beta-ucb $beta
    elif [ "$agt_index" -eq 2 ]; then
      echo "Running gp-sinc"
      python run_agent.py --agent "gp-sinc" --num-trials 10 --func $func --data-dir $dest_dir --gamma 0.0 --beta-ucb $beta
    elif [ "$agt_index" -eq 3 ]; then
        echo "Running gp-matern"
        python run_agent.py --agent "gp-matern" --num-trials 10 --func $func --data-dir $dest_dir --gamma 0.0 --beta-ucb $beta
    fi
done
