#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=0-6:00
#SBATCH --array=0-2

module load python/3.10 scipy-stack
source ~/projects/def-celiasmi/ns2dumon/mcbo-env/bin/activate

run_index=$(( SLURM_ARRAY_TASK_ID / 5 ))
func_index=$(( SLURM_ARRAY_TASK_ID % 5 ))

sspdim=97
dest_dir="data"
funcs=("past" "rna_inverse_fold" "mig_optimization")
func="${funcs[$func_index]}"
echo "Fucntion: $func"
if [ "$run_index" -eq 0 ]; then
    echo "Running default"
    python run_agent_mcbo.py --task-id $func --num-trials 30 --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb 30.
elif [ "$run_index" -eq 1 ]; then
    echo "Running loihi-sim"
    python run_agent_mcbo.py --task-id $func --num-trials 30 --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb 30. --nengo --backend "loihi-sim"
fi