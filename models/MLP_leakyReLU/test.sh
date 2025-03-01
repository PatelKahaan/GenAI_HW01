#!/bin/bash

#BSUB -n 1

#BSUB -W 24:00

#BSUB -q gpu

#BSUB -gpu "num=1:mode=shared:mps=no"

#BSUB -o /share/csc591s25/kpatel48/logs/test.out.%J

#BSUB -e /share/csc591s25/kpatel48/logs/test.err.%J



# Load environment

source ~/.bashrc

conda activate /share/csc591s25/conda_env/new_env



# Change to the directory containing the MLP model code

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/MLP"



# Run training

python3 -m cache_replacement.policy_learning.cache_model.main --experiment_base_dir=/share/csc591s25/kpatel48/tmp --experiment_name=test --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" --model_configs="cache_replacement/policy_learning/cache_model/configs/default.json" --model_bindings="lstm_hidden_size=128" --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv" --batch_size=64 --total_steps=20000

