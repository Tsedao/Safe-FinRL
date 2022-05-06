#!/bin/bash

# find nsteps length
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=64 --look_back=10 > steps_log/log64.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=16 --look_back=10 > steps_log/log16.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=4 --look_back=10 > steps_log/log4.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=1 --look_back=10 > steps_log/log1.out &

# find trace type
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 > trace_log/log_re.out &
nohup python trace_main.py --trace_type='qlambda' --lambda_=0.8 --nsteps=30 --look_back=10 > trace_log/log_qlam.out &
nohup python trace_main.py --trace_type='qlambda' --lambda_=1 --nsteps=30 --look_back=10 > trace_log/log_uncorrect.out &
nohup python trace_main.py --trace_type='treebackup' --lambda_=0.8 --nsteps=30 --look_back=10 > trace_log/log_tree.out &
nohup python trace_main.py --trace_type='IS' --lambda_=0.8 --nsteps=30 --look_back=10 > trace_log/log_is.out &

# find model type
nohup python trace_main.py --trace_type='retrace' --model_type='lstm' --lambda_=0.8 --nsteps=30 --look_back=10 > model_log/log_lstm.out &
nohup python trace_main.py --trace_type='retrace' --model_type='transformer' --lr_actor=0.00005 --lr_critic=0.00003 --lr_alpha=0.0003 --lambda_=0.8 --nsteps=30 --look_back=10 > model_log/log_trans.out &
nohup python trace_main.py --trace_type='retrace' --model_type='fc' --lambda_=0.8 --nsteps=30 --look_back=10 > model_log/log_fc.out &

# test on different environment
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 --env=0 > env_log/log_e0.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 --env=1 > env_log/log_e1.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 --env=2 > env_log/log_e2.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 --env=3 > env_log/log_e3.out &

# test on lambda
nohup python trace_main.py --trace_type='retrace' --lambda_=0.5 --nsteps=30 --look_back=10 > lambda_log/log5.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.6 --nsteps=30 --look_back=10 > lambda_log/log6.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.7 --nsteps=30 --look_back=10 > lambda_log/log7.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=30 --look_back=10 > lambda_log/log8.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=0.9 --nsteps=30 --look_back=10 > lambda_log/log9.out &
nohup python trace_main.py --trace_type='retrace' --lambda_=1 --nsteps=30 --look_back=10 > lambda_log/log10.out &


echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
