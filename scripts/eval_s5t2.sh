#!/bin/bash

experiment_name="LSTM_byte_embedder"
BASE_DIR="/share/csc591s25/kpatel48/tmp"

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/LSTM"

checkpoint_path="${BASE_DIR}/${experiment_name}/checkpoints/20000.ckpt"
config_path="${BASE_DIR}/${experiment_name}/model_config.json"

echo "Running evaluation for ${experiment_name}..."

if [ ! -d "${BASE_DIR}/${experiment_name}" ]; then
    echo "Error: Directory ${BASE_DIR}/${experiment_name} not found! Skipping."
    exit 1
fi

python3 -m cache_replacement.policy_learning.cache.main \
    --experiment_base_dir="/share/csc591s25/kpatel48/tmp_eval" \
    --experiment_name="${experiment_name}" \
    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
    --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
    --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_313B_test.csv" \
    --config_bindings="eviction_policy.scorer.checkpoint=\"${checkpoint_path}\"" \
    --config_bindings="eviction_policy.scorer.config_path=\"${config_path}\"" \
    --warmup_period=0

echo "Evaluation for ${experiment_name} completed."
echo "All evaluations completed!"
