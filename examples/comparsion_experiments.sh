python3 run_agent.py --func himmelblau --agent gp-mi
python3 run_agent.py --func himmelblau --agent ssp-mi
python3 run_agent.py --func himmelblau --agent hex-mi

python3 run_agent.py --func branin-hoo --agent gp-mi
python3 run_agent.py --func branin-hoo --agent ssp-mi
python3 run_agent.py --func branin-hoo --agent hex-mi

python3 run_agent.py --func goldstein-price --agent gp-mi
python3 run_agent.py --func goldstein-price --agent ssp-mi
python3 run_agent.py --func goldstein-price --agent hex-mi

python3 run_agent.py --func mccormick --agent gp-mi
python3 run_agent.py --func mccormick --agent ssp-mi
python3 run_agent.py --func mccormick --agent hex-mi

python3 run_agent.py --func beale --agent gp-mi
python3 run_agent.py --func beale --agent ssp-mi
python3 run_agent.py --func beale --agent hex-mi

python3 run_agent.py --func styblinski-tang3 --agent gp-mi
python3 run_agent.py --func styblinski-tang3 --agent ssp-mi
python3 run_agent.py --func styblinski-tang3 --agent hex-mi

python3 plot-results.py --path data/
