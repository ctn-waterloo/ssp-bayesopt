#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=0-6:00
#SBATCH --array=0-9

module load python/3.10 scipy-stack
source ~/projects/def-celiasmi/ns2dumon/bosspenv/bin/activate

run_index=$(( SLURM_ARRAY_TASK_ID / 3 ))
func_index=$(( SLURM_ARRAY_TASK_ID % 3 ))

beta=0.1
sspdim=97
dest_dir="data"
funcs=("branin-hoo" "himmelblau" "goldstein-price")
func="${funcs[$func_index]}"

echo "Function: $func"
echo "Running ssp-hex-loihi"
python run_agent.py --agent "ssp-hex" --num-trials 1 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb $beta --nengo --backend "loihi-sim"

