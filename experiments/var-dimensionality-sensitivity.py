"""Dimensionality sensitivity sweep: run GP and SSP-rand agents across input dims.

Dispatches multiple run_agent.py calls via os.system() for dims 1..9 and a
range of SSP dimensions. Results are saved to --data-dir/<dim>/<ssp_dim>/.

Usage:
    python var-dimensionality-sensitivity.py --data-dir /path/to/results
"""
import os
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default=os.path.join(os.getcwd(), 'data', 'var-dim-sensitivity'))
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=10)
    parser.add_argument('--func', dest='func_name', type=str, default='styblinski-tang')
    args = parser.parse_args()

    func_name = args.func_name
    dims = np.arange(1, 10, 1)
    ssp_dims = 1 + 2 ** np.array([5, 6, 7, 8, 9, 10])
    num_trials = args.num_trials

    for dim in dims:
        dest_dir = os.path.join(args.data_dir, str(dim))
        for gp_agt in ['gp-matern', 'gp-sinc']:
            os.system(
                f"python run_agent.py --agent {gp_agt}"
                f" --num-trials {num_trials} --func {func_name}{dim}"
                f" --data-dir {dest_dir} --len-scale -1"
            )
        for ssp_dim in ssp_dims:
            ssp_dest_dir = os.path.join(dest_dir, str(ssp_dim))
            for agt in ['ssp-rand']:
                os.system(
                    f"python run_agent.py --agent {agt}"
                    f" --num-trials {num_trials} --func {func_name}{dim}"
                    f" --data-dir {ssp_dest_dir} --len-scale -1 --ssp-dim {ssp_dim}"
                )
