#!/bin/bash

sizes=(32 64 128 256 512)
BASE_DIR="/share/csc591s25/kpatel48/tmp"

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/MLP"

for size in "${sizes[@]}"; do
    experiment_name="mlp_width_${size}"
    checkpoint_path="${BASE_DIR}/${experiment_name}/checkpoints/20000.ckpt"
    config_path="${BASE_DIR}/${experiment_name}/model_config.json"

    echo "Running evaluation for ${experiment_name}..."

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
