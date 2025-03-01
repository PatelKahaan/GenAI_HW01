#!/bin/bash

history_lengths=(20 40 60 80 100 120 140)
BASE_DIR="/share/csc591s25/kpatel48/tmp"

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/RNN_with_Attention"

for history_length in "${history_lengths[@]}"; do
    experiment_name="rnn_w_att_seq_${history_length}"
    checkpoint_path="${BASE_DIR}/${experiment_name}/checkpoints/20000.ckpt"
    config_path="${BASE_DIR}/${experiment_name}/model_config.json"

    echo "Running evaluation for ${experiment_name} with sequence length ${history_length}..."

    if [ ! -d "${BASE_DIR}/${experiment_name}" ]; then
        echo "Error: Directory ${BASE_DIR}/${experiment_name} not found! Skipping."
        continue
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
done

echo "All evaluations completed!"