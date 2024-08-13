python fit.py
python sample.py

python estimate.py
python evaluate.py
python present_results.py

python estimate.py -c configs/estimation_minimal.yml
python evaluate.py -c configs/estimation_minimal.yml
python present_results.py -ec configs/estimation_minimal.yml

# Added after the initial submission 

#python estimate.py -c configs/estimation_dml.yml
#python evaluate.py -c configs/estimation_dml.yml
#python present_results.py -ec configs/estimation_dml.yml

# For short horizons

python sample.py -c configs/simulator_short.yml
python estimate.py -c configs/estimation_short.yml
python evaluate.py -c configs/estimation_short.yml
python present_results.py -sc configs/simulator_short.yml -ec configs/estimation_short.yml