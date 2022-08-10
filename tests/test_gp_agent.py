import numpy as np
import functions


from ssp_bayes_opt import BayesianOptimization 


# def test_gp_init():
#     func, bounds, T = functions.factory('branin-hoo')
#     bo = BayesianOptimization(f=func, bounds=bounds)
#     agt, init_xs, init_ys = bo.initialize_agent(agent_type='gp')
#     print(np.exp(agt.gp.kernel_.theta))


# def test_gp_update():
#     func, bounds, T = functions.factory('branin-hoo')
#     bo = BayesianOptimization(f=func, bounds=bounds)
#     agt, init_xs, init_ys = bo.initialize_agent(agent_type='gp')
#     print(np.exp(agt.gp.kernel_.theta))

def test_static_gp_init():
    func, bounds, T = functions.factory('branin-hoo')
    bo = BayesianOptimization(f=func, bounds=bounds)
    agt, init_xs, init_ys = bo.initialize_agent(agent_type='static-gp')
    print('lenscale', agt.gp.kernel_.length_scale)

# def test_static_gp_update():
#     func, bounds, T = functions.factory('branin-hoo')
#     bo = BayesianOptimization(f=func, bounds=bounds)
#     agt, init_xs, init_ys = bo.initialize_agent(agent_type='static-gp')
#     print(np.exp(agt.gp.kernel_.theta))
