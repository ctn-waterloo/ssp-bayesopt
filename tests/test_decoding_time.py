import numpy as np
import time

import functions

from ssp_bayes_opt import sspspace

def time_decode(ssp, space, method='from-set'):
    start = time.thread_time_ns()
#     x = space.decode(ssp, method=method,num_samples=300**2)
    x = space.decode(ssp, method=method)
    end = time.thread_time_ns() - start
    return end


def preallocate_time_decode(ssp, space, method='from-set', decoding_method='grid'):
    init_samples = space.get_sample_pts_and_ssps(300**2, decoding_method)
    start = time.thread_time_ns()
    x = space.decode(ssp, method=method,samples=init_samples)
    end = time.thread_time_ns() - start
    return end

def get_timing(func_name):
    # Get func info
    func, bounds, T = functions.factory(func_name)
    # Make hex space
    hex_space = sspspace.HexagonalSSPSpace(
                    bounds.shape[0],
                    ssp_dim=151, 
                    domain_bounds=bounds, 
                    length_scale=4)

    # Make rand space
    rand_space = sspspace.RandomSSPSpace(
                    bounds.shape[0],
                    ssp_dim=151, 
                    domain_bounds=bounds, 
                    length_scale=4)
    # encode point 
    hex_origin = hex_space.encode(np.array([[0,0]]))
#     hex_timing = time_decode(hex_origin, hex_space, method='direct-optim')
    hex_timing = preallocate_time_decode(hex_origin, hex_space, 
            method='direct-optim',
            decoding_method='length-scale')

    rand_origin = rand_space.encode(np.array([[0,0]]))
#     rand_timing = time_decode(rand_origin, rand_space, method='direct-optim')
    rand_timing = preallocate_time_decode(rand_origin, rand_space, 
            method='direct-optim',
            decoding_method='length-scale')
    return hex_timing*1e-9, rand_timing*1e-9

funcs = ['branin-hoo', 'himmelblau', 'goldstein-price']

np.random.seed(0)
num_trials = 10
for f in funcs:
    print(f)
    timings = [get_timing(f) for _ in range(num_trials)]
    mu_timings = np.mean(timings, axis=0)
    std_timings = np.std(timings, axis=0)
    print(f'\t hex: {mu_timings[0]} +- {std_timings[0]}')
    print(f'\t rand: {mu_timings[1]} +- {std_timings[1]}')
