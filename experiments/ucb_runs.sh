#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=0-4:00
#SBATCH --array=0-15

module load python/3.10 scipy-stack
source ~/projects/def-celiasmi/ns2dumon/bosspenv/bin/activate

func_index=$(( SLURM_ARRAY_TASK_ID % 3 ))
run_index=$(( SLURM_ARRAY_TASK_ID / 3 ))

beta=1
sspdim=97
dest_dir="./data2"
funcs=("branin-hoo" "himmelblau" "goldstein-price")
func="${funcs[$func_index]}"

if [ "$run_index" -eq 0 ]; then
    echo "Running ssp-rand"
    python run_agent.py --agent "ssp-rand" --num-trials 30 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb $beta
elif [ "$run_index" -eq 1 ]; then
    echo "Running ssp-hex"
    python run_agent.py --agent "ssp-hex" --num-trials 30 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb $beta
elif [ "$run_index" -eq 2 ]; then
    echo "Running gp-sinc"
    python run_agent.py --agent "gp-sinc" --num-trials 30 --func $func --data-dir $dest_dir --gamma 0.0 --beta-ucb $beta
elif [ "$run_index" -eq 3 ]; then
    echo "Running gp-matern"
    python run_agent.py --agent "gp-matern" --num-trials 30 --func $func --data-dir $dest_dir --gamma 0.0 --beta-ucb $beta
elif [ "$run_index" -eq 4 ]; then
    echo "Running rff"
    python run_agent.py --agent "rff" --ssp-dim $sspdim --num-trials 30 --func $func --data-dir $dest_dir --gamma 0.0 --beta-ucb $beta
elif [ "$run_index" -eq 5 ]; then
    echo "Running ssp-hex-loihi"
    #python run_agent.py --agent "ssp-hex" --num-trials 10 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim $sspdim --gamma 0.0 --beta-ucb $beta --nengo --backend "loihi-sim" --num-neurons 8
else
    echo "Unexpected run_index value: $run_index"
fi
